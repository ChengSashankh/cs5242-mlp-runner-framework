import os
from datetime import datetime


class Logger:
    def __init__(self, output_dir, file_name, class_name):
        self.output_dir = output_dir
        self.file_name = file_name
        self.class_name = class_name
        self.fp = open(os.path.join(self.output_dir, file_name), "a+")

    def log(self, to_log):
        now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        self.fp.write(f"{now}: {to_log}\n")
        self.fp.flush()

    def close(self):
        self.fp.close()


