from . import analysis, live_cmd, ml_cmd, portfolio, scanner, scout_cmd, setup, sim_cmd, trading

MODULES = [portfolio, trading, scanner, analysis, ml_cmd, sim_cmd, live_cmd, setup, scout_cmd]


def build_parser_and_handlers(subparsers):
    handlers = {}
    for mod in MODULES:
        handlers.update(mod.register(subparsers))
    return handlers
