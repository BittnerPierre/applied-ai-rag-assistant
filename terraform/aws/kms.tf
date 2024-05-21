resource "aws_kms_key" "key" {
  description             = "Ai Assistant ECS cluster CloudWatch log KMS key"
  deletion_window_in_days = 7
}
