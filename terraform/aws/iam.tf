resource "aws_iam_role" "ai_assistant_ecs_execution_role" {
  name = "ai_assistant_ecs_execution_role_https"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Action = "sts:AssumeRole",
      Effect = "Allow",
      Principal = {
        Service = "ecs-tasks.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ai_assistant_secret_read_role_attachment" {
  policy_arn = "arn:aws:iam::aws:policy/SecretsManagerReadWrite"
  role       = aws_iam_role.ai_assistant_ecs_execution_role.name
}

resource "aws_iam_role_policy_attachment" "ai_assistant_cloud_watch_access_role_attachment" {
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchLogsFullAccess"
  role       = aws_iam_role.ai_assistant_ecs_execution_role.name
}

resource "aws_iam_role_policy_attachment" "ai_assistant_efs_access_role_attachment" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonElasticFileSystemFullAccess"
  role       = aws_iam_role.ai_assistant_ecs_execution_role.name
}

resource "aws_iam_role_policy_attachment" "ai_assistant_ecs_execution_role_attachment" {
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
  role       = aws_iam_role.ai_assistant_ecs_execution_role.name
}

resource "aws_iam_role_policy_attachment" "ai_assistant_ec2_container_registry_role_attachment" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
  role       = aws_iam_role.ai_assistant_ecs_execution_role.name
}

resource "aws_iam_role_policy_attachment" "ai_assistant_bedrock_access_role_attachment" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonBedrockFullAccess"
  role       =  aws_iam_role.ai_assistant_ecs_execution_role.name
}