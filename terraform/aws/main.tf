# data "aws_vpc" "ai_assistant_vpc" {
#   id = "vpc-72dfcf14"
# }

# data  "aws_subnet" "ai_assistant_subnet_1" {
#   id = "subnet-0a75fa50"
# }

# data  "aws_subnet" "ai_assistant_subnet_2" {
#   id = "subnet-1fd38e57"
# }

# data  "aws_subnet" "ai_assistant_subnet_3" {
#   id = "subnet-703c7116"
# }

data "aws_vpc" "ai_assistant_vpc" {
  id = var.vpc_id
}

data "aws_subnet" "ai_assistant_subnet_1" {
  id = var.subnet_id_1
}

data "aws_subnet" "ai_assistant_subnet_2" {
  id = var.subnet_id_2
}

data "aws_subnet" "ai_assistant_subnet_3" {
  id = var.subnet_id_3
}

resource "aws_kms_key" "key" {
  description             = "key"
  deletion_window_in_days = 7
}

resource "aws_ecs_cluster" "ai_assistant_cluster" {
  name = "ai_assistant"
  configuration {
      execute_command_configuration {
        kms_key_id = aws_kms_key.key.arn
        logging    = "OVERRIDE"

        log_configuration {
          cloud_watch_encryption_enabled = true
          cloud_watch_log_group_name     = aws_cloudwatch_log_group.ai_assistant-cloudwatch-log.name
        }
      }
    }
}

resource "aws_ecs_cluster_capacity_providers" "cluster" {
  cluster_name = aws_ecs_cluster.ai_assistant_cluster.name

  capacity_providers = ["FARGATE", "FARGATE_SPOT"]

  default_capacity_provider_strategy {
    capacity_provider = "FARGATE_SPOT"
  }
}

resource "aws_cloudwatch_log_group" "ai_assistant-cloudwatch-log" {
  name = "/ecs/ai_assistant-taskdef-iac-https"
}
resource "aws_security_group" "ai_assistant_security_group" {
  name        = "ai_assistant-security-group-https"
  vpc_id      = data.aws_vpc.ai_assistant_vpc.id
  
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "TCP"
    cidr_blocks = ["0.0.0.0/0"]
  }
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "TCP"
    cidr_blocks = ["0.0.0.0/0"]
  }
  egress {
    from_port        = 0
    to_port          = 0
    protocol         = "-1"
    cidr_blocks      = ["0.0.0.0/0"]
    ipv6_cidr_blocks = ["::/0"]
  }
}


resource "aws_ecs_service" "ai_assistant_service" {
  name            = "ai_assistant-service-iac-https"
  cluster         = aws_ecs_cluster.ai_assistant_cluster.id
  task_definition = aws_ecs_task_definition.ai_assistant_task_definition.arn
  force_new_deployment = true
  capacity_provider_strategy {
    capacity_provider =  "FARGATE_SPOT"
    base = 1
    weight            = 1
  }
  network_configuration {
    subnets = [data.aws_subnet.ai_assistant_subnet_1.id, data.aws_subnet.ai_assistant_subnet_2.id, data.aws_subnet.ai_assistant_subnet_3.id]
    security_groups = [aws_security_group.ai_assistant_security_group.id]
    assign_public_ip = true
  }
  deployment_circuit_breaker {
    enable = true
    rollback = true
  }
  desired_count = 1

  load_balancer {
    target_group_arn = aws_lb_target_group.ai_assistant_target_group_https.arn
    container_name   = "ai_assistant_https"
    container_port   = 80
  }
  
  depends_on = [aws_lb_listener.application_lb_listener]
}

resource "aws_ecs_task_definition" "ai_assistant_task_definition" {
  family                   = "ai_assistant-taskdef-iac-https"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]

  cpu          = "512"
  memory       = "1024"

  execution_role_arn =  data.aws_iam_role.ecs_execution_role.arn
  task_role_arn =  data.aws_iam_role.ecs_execution_role.arn

  container_definitions = jsonencode([{
    name  = "ai_assistant_https"
    image = "441525731509.dkr.ecr.eu-west-1.amazonaws.com/ai_assistant:latest"
    cpu = 512
    memory = 1024
    runtime_platform = {
        "cpuArchitecture": "X86_64",
        "operatingSystemFamily": "LINUX"
    }
    memoryReservation=  1024
    portMappings = [{
      name = "ai_assistant-80-tcp"
      containerPort = 80,
      hostPort      = 80
      appProtocol = "http"
    }]
    logConfiguration = {
                logDriver= "awslogs"
                options = {
                    awslogs-create-group="true"
                    awslogs-group= "/ecs/ai_assistant-taskdef-iac"
                    awslogs-region= "eu-west-1"
                    awslogs-stream-prefix= "ecs"
                },
            }
    essential = true
    secrets = [
        {
          "name": "OPENAI_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:eu-west-1:441525731509:secret:ai-assistant-G3Ft3L:OPENAI_API_KEY::"
        }
      ]
  }])
}

data "aws_iam_role" "ecs_execution_role" {
  name = "ecs_execution_role_https"
}
# resource "aws_iam_role" "ecs_execution_role" {
#   name = "ecs_execution_role_https"

#   assume_role_policy = jsonencode({
#     Version = "2012-10-17",
#     Statement = [{
#       Action = "sts:AssumeRole",
#       Effect = "Allow",
#       Principal = {
#         Service = "ecs-tasks.amazonaws.com"
#       }
#     }]
#   })
# }

resource "aws_iam_role_policy_attachment" "secret_read_role_attachment" {
  policy_arn = "arn:aws:iam::aws:policy/SecretsManagerReadWrite"
  role       = data.aws_iam_role.ecs_execution_role.name
}

resource "aws_iam_role_policy_attachment" "cloud_watch_access_role_attachment" {
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchLogsFullAccess"
  role       =  data.aws_iam_role.ecs_execution_role.name
}

resource "aws_iam_role_policy_attachment" "ecs_execution_role_attachment" {
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
  role       =  data.aws_iam_role.ecs_execution_role.name
}

resource "aws_iam_role_policy_attachment" "ec2_container_registry_role_attachment" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
  role       =  data.aws_iam_role.ecs_execution_role.name
}

resource "aws_lb_target_group" "ai_assistant_target_group_https" {
  name     = "ai-assistant-group-https"
  port     = 80
  protocol = "HTTP"
  vpc_id   = data.aws_vpc.ai_assistant_vpc.id
  target_type = "ip"
}

resource "aws_lb" "application_lb" {
  name = "ai-assistant-alb-tf-https"
  internal = false
  load_balancer_type = "application"
  security_groups = [aws_security_group.ai_assistant_security_group.id]
  subnets = [data.aws_subnet.ai_assistant_subnet_1.id, data.aws_subnet.ai_assistant_subnet_2.id, data.aws_subnet.ai_assistant_subnet_3.id]
}

resource "aws_lb_listener" "application_lb_listener" {
  load_balancer_arn = aws_lb.application_lb.arn
  port              = 443
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-2016-08"

  certificate_arn = data.aws_acm_certificate.ai_assistant_certificate.arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.ai_assistant_target_group_https.arn
  }
}

resource "aws_lb_listener" "application_lb_listener_redirect" {
  load_balancer_arn = aws_lb.application_lb.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type = "redirect"

    redirect {
      port        = "443"
      protocol    = "HTTPS"
      status_code = "HTTP_301"
    }
  }
}

data "aws_acm_certificate" "ai_assistant_certificate" {
  domain       = var.dns_url
}

data "aws_route53_zone" "ai_assistant_zone" {
  name = var.dns_url
}

resource "aws_route53_record" "example_record" {
  zone_id = data.aws_route53_zone.ai_assistant_zone.zone_id
  name    = var.dns_url
  type    = "A"
  alias {
    name                   = aws_lb.application_lb.dns_name
    zone_id                = aws_lb.application_lb.zone_id
    evaluate_target_health = true
  }
}
