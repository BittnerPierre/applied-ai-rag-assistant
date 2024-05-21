data "aws_acm_certificate" "ai_assistant_certificate" {
  domain = var.dns_url_app_subnet
}

data "aws_route53_zone" "ai_assistant_zone" {
  name = var.dns_url
}

resource "aws_route53_record" "record" {
  zone_id = data.aws_route53_zone.ai_assistant_zone.zone_id
  name    = var.dns_url_app_subnet
  type    = "A"
  alias {
    name                   = aws_lb.application_lb.dns_name
    zone_id                = aws_lb.application_lb.zone_id
    evaluate_target_health = true
  }
}
