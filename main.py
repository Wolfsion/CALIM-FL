import argparse
from custom_path import auto_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-y', '--yaml_path',
                        help='config file',
                        dest='config',
                        type=str,
                        default='e1')
    yml_arg = parser.parse_args()
    auto_config(yml_arg.config)

    from dl.test_unit import main as dl_main
    from utils.test_unit import main as statistics_main
    dl_main()
    # statistics_main()
