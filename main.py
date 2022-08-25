import argparse
from custom_path import auto_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-y', '--yaml_path',
                        help='config file',
                        dest='config',
                        type=str,
                        default='e1')
    parser.add_argument('-s', '--federal_simulation',
                        help='Federal simulation is True, the single is False.',
                        dest='env',
                        type=bool,
                        default=True)
    yml_arg = parser.parse_args()
    auto_config(yml_arg.config)

    from dl.test_unit import main as dl_main
    from federal.test_unit import main as fl_main

    if yml_arg.env:
        fl_main()
    else:
        dl_main()

    # from utils.test_unit import main as statistics_main
    # # statistics_main()
