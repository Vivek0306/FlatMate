# spinner.py

import itertools
import sys
import time

def spinner(stop_event):
    for char in itertools.cycle(["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]):
        if stop_event.is_set():
            break
        sys.stdout.write(f"\r🤖 Thinking {char}")
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write("\r" + " " * 30 + "\r")
