import argparse

parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument('--user', '-u',
                    type=str,
                    default='tsaoni',
                    help='username')
parent_parser.add_argument('--debug', default=False, required=False,
                        action='store_true', dest="debug", help='debug flag')
main_parser = argparse.ArgumentParser()
service_subparsers = main_parser.add_subparsers(title="service",
                    dest="service_command")
service_parser = service_subparsers.add_parser("first", help="first",
                    parents=[parent_parser])
service_parser.add_argument('--electric', type=str, default="100", 
                          help='debug flag')
action_subparser = service_parser.add_subparsers(title="action",
                    dest="action_command")
action_parser = action_subparser.add_parser("second", help="second",
                    parents=[parent_parser])
action_parser.add_argument('act', type=str, default="use", 
                          help='debug flag')

args = main_parser.parse_args()
