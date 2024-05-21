resource "aws_lb_target_group" "ai_assistant_target_group_https" {
  name        = "ai-assistant-group-https"
  port        = 80
  protocol    = "HTTP"
  vpc_id      = data.aws_vpc.ai_assistant_vpc.id
  target_type = "ip"
}

resource "aws_lb" "application_lb" {
  name               = "ai-assistant-alb-tf-https"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.ai_assistant_security_group.id]
  subnets            = [data.aws_subnet.ai_assistant_subnet_1.id, data.aws_subnet.ai_assistant_subnet_2.id, data.aws_subnet.ai_assistant_subnet_3.id]
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
