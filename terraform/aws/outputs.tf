output "vpc_id" {
  description = "VPC ID"
  value       = var.vpc_id
}

output "subnet_1_id" {
  description = "Subnet 1 ID"
  value       = var.subnet_id_1
}

output "subnet_2_id" {
  description = "Subnet 2 ID"
  value       = var.subnet_id_2
}

output "subnet_3_id" {
  description = "Subnet 3 ID"
  value       = var.subnet_id_3
}

output "lb_arn" {
  description = "Load Balancer arn"
  value       = aws_lb.application_lb.arn
}

output "lb_dns_name" {
  description = "Load Balancer DNS name"
  value       = aws_lb.application_lb.dns_name
}

output "aws_ecs_cluster_arn" {
  description = "ECS Cluster arn"
  value       = aws_ecs_cluster.ai_assistant_cluster.id
}

output "app_url" {
  description = "URL to access the deployed application"
  value       = var.dns_url_app_subnet
}

output "ecr_image_url" {
  description = "URL to the image in ECR repository"
  value       = var.ecr_image_url
}

output "cloudwatch_name" {
  description = "CloudWatch log group name"
  value       = aws_cloudwatch_log_group.ai_assistant-cloudwatch-log.name
}

output "cloudwatch_arn" {
  description = "CloudWatch log group arn"
  value       = aws_cloudwatch_log_group.ai_assistant-cloudwatch-log.arn
}
