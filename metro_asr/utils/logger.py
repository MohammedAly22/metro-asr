import sys
import logging
from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

_THEME = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
})


def _utf8_stdout():
    """
    Force UTF-8 on stdout.

    Windows consoles default to a legacy code page (cp1252), which raises
    UnicodeEncodeError the moment an Arabic transcript is logged — which this
    trainer does on every eval step.
    """
    stream = sys.stdout
    if getattr(stream, "encoding", "").lower().replace("-", "") not in ("utf8",):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    return stream


_console = Console(theme=_THEME, file=_utf8_stdout())
_logger_cache = {}


def get_logger(name="metro-asr", level=logging.INFO):
    if name in _logger_cache:
        return _logger_cache[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    handler = RichHandler(
        console=_console,
        show_time=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
    )
    handler.setLevel(level)
    fmt = logging.Formatter("%(message)s", datefmt="[%X]")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.propagate = False

    _logger_cache[name] = logger
    return logger


def print_banner():
    banner = """
[bold cyan]
 ███╗   ███╗███████╗████████╗██████╗  ██████╗        █████╗ ███████╗██████╗
 ████╗ ████║██╔════╝╚══██╔══╝██╔══██╗██╔═══██╗      ██╔══██╗██╔════╝██╔══██╗
 ██╔████╔██║█████╗     ██║   ██████╔╝██║   ██║█████╗███████║███████╗██████╔╝
 ██║╚██╔╝██║██╔══╝     ██║   ██╔══██╗██║   ██║╚════╝██╔══██║╚════██║██╔══██╗
 ██║ ╚═╝ ██║███████╗   ██║   ██║  ██║╚██████╔╝      ██║  ██║███████║██║  ██║
 ╚═╝     ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝       ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝
[/bold cyan]
[dim]Egyptian Arabic + Code-Switching ASR | Non-Autoregressive CTC[/dim]
"""
    _console.print(banner)


def print_config_summary(config):
    from rich.table import Table

    table = Table(title="⚙️  Configuration Summary", show_header=True, header_style="bold magenta")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    m = config["model"]
    table.add_row("Model Name", m["name"])
    table.add_row("Hidden Dim", str(m["d_model"]))
    table.add_row("Layers", str(m["n_layers"]))
    table.add_row("Attention Heads", str(m["n_heads"]))
    table.add_row("FF Multiplier", str(m["ff_multiplier"]))
    table.add_row("Conv Kernel", str(m["conv_kernel_size"]))
    table.add_row("Dropout", str(m["dropout"]))

    t = config["training"]
    table.add_row("Batch Size", str(t["batch_size"]))
    table.add_row("Learning Rate", str(t["learning_rate"]))
    table.add_row("Max Steps", str(t["max_steps"]))
    table.add_row("Warmup Steps", str(t["warmup_steps"]))
    table.add_row("FP16", str(t.get("fp16", False)))

    table.add_row("Tokenizer", config["tokenizer"]["type"])
    table.add_row("Vocab Size", str(config["tokenizer"]["vocab_size"]))

    _console.print(table)
