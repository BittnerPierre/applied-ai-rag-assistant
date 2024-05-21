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

data "aws_secretsmanager_secret" "secret" {
  name = var.secret_name
}
