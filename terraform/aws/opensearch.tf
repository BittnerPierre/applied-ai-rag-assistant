resource "aws_opensearch_domain" "ai_assistant_opensearch_domain" {
  domain_name    = var.opensearch_domain_name
  engine_version = "OpenSearch_2.13"

  cluster_config {
    instance_type          = "r5.large.search"
    zone_awareness_enabled = false
    instance_count = 1
    multi_az_with_standby_enabled = false
  }

    ebs_options {
    ebs_enabled = true
    volume_size = 10
  }

  vpc_options {
    subnet_ids = [
      data.aws_subnet.ai_assistant_subnet_1.id
    ]

    security_group_ids = [aws_security_group.ai_assistant_opensearch.id]
  }
}
