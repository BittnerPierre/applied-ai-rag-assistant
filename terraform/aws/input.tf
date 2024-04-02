variable "vpc_id" {
  description = "The ID of the VPC"
  default = "vpc-72dfcf14"
  type        = string
}

variable "subnet_id_1" {
  description = "The ID of the first subnet"
  default = "subnet-0a75fa50"
  type        = string
}

variable "subnet_id_2" {
  description = "The ID of the second subnet"
  default = "subnet-1fd38e57"
  type        = string
}

variable "subnet_id_3" {
  description = "The ID of the third subnet"
  default = "subnet-703c7116"
  type        = string
}

variable "dns_url" {
  description = "The base DNS url (without subnet)."
  default = "lab-finaxys.net"
  type        = string
}

variable "dns_url_app_subnet" {
  description = "The DNS URL of your application. (It need to have a valid HTTPS certificate and a route 53 hosted zone)"
  default = "ai-assistant.lab-finaxys.net"
  type        = string
}