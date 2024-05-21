
resource "aws_kms_key" "volume_key" {
  description             = "Ai Assistant EFS Volume key"
  deletion_window_in_days = 7
}

resource "aws_efs_file_system" "ai_assistant_efs_file_system" {
  encrypted  = true
  kms_key_id = aws_kms_key.volume_key.arn

  tags = {
    Name = "ai-assistant"
  }
}

resource "aws_efs_mount_target" "ai_assistant_efs_mount_target_1" {
  file_system_id  = aws_efs_file_system.ai_assistant_efs_file_system.id
  subnet_id       = data.aws_subnet.ai_assistant_subnet_1.id
  security_groups = [aws_security_group.ai_assistant_security_group.id]
}

resource "aws_efs_mount_target" "ai_assistant_efs_mount_target_2" {
  file_system_id  = aws_efs_file_system.ai_assistant_efs_file_system.id
  subnet_id       = data.aws_subnet.ai_assistant_subnet_2.id
  security_groups = [aws_security_group.ai_assistant_security_group.id]
}

resource "aws_efs_mount_target" "ai_assistant_efs_mount_target_3" {
  file_system_id  = aws_efs_file_system.ai_assistant_efs_file_system.id
  subnet_id       = data.aws_subnet.ai_assistant_subnet_3.id
  security_groups = [aws_security_group.ai_assistant_security_group.id]
}
