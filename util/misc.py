import os
import pygame
import traceback
import threading

def crash_handler(*exc_info):
    exc_info = "".join(traceback.format_exception(*exc_info))
    log("app crashed (lmao)\n")
    log(exc_info)
    pygame.display.message_box(title="uhhhhhhhhhhhhh", message=f"I just downvoted your comment.\n\nFAQ:\n\nWhat does this mean?\n\n{exc_info}", message_type="error")

class Logging:
    """very simple hack to get messages to print to console and to a file at the same time, 
    less effort than the logging library"""
    init = False
    lock = threading.Lock()
    
    def log(message, print_out=True):
        with Logging.lock:
            if print_out:
                print(message)
            if not Logging.init:
                Logging.init = True
                try:
                    os.remove("assets/log.txt")
                except FileNotFoundError:
                    pass
            with open("assets/log.txt", "a", encoding="UTF-8") as logfile:
                logfile.write(message + "\n")

def log(message : str, print_out=True):
    """alias for Logging.log(message),
    prints a message and writes it to assets/log.txt at the same time"""
    Logging.log(message, print_out)

class Settings:
    def __init__(self):
        self.val_tests = {}
        self.descriptions = {}

    def add(self, setting, default, val_test, description):
        """add a new setting with a default value that uses python shortcuts to be accessed by Settings.setting directly
        args:
        setting: name of the setting
        default: default value
        val_test: function that evaluates to True if the value for this setting read from settings.ini is valid
        description: description of the setting to be written to settings.ini"""
        self.val_tests[setting] = val_test
        self.descriptions[setting] = description
        setattr(self, setting, default)

    def verify(self, setting, value):
        """returns value if the value is valid for setting, else the default value of the setting"""
        try:
            value = eval(value)
            valid = self.val_tests[setting](value)
        except:
            valid = False
        if not valid: 
            log(f"Value '{value}' is not valid for {setting}, setting to default of {self.__dict__[setting]}...")
        return value if valid else self.__dict__[setting] 

    def read(self):
        """reads settings.ini and sets the settings accordingly
        \n default settings need to be added first with .add"""
        with open("settings.ini") as file:
            for line in file.readlines():
                if not (line := line.strip("\n")): continue #empty line
                if line.startswith("#"): continue #comment
                if len(setting := line.split("=")) != 2: continue #can't parse setting
                setting, value = [token.strip() for token in setting]
                if setting not in self.descriptions: continue #setting doesn't exist 
                setattr(self, setting, self.verify(setting, value))

    def update(self, setting, value):
        """attempts to change setting to value, doesn't change if new value is invalid"""
        setattr(self, setting, self.verify(setting, value))

    def save(self):
        """save the settings added with .add with the values modified by .read & .update"""
        with open("settings.ini", "w") as file:
            for setting, description in self.descriptions.items():
                file.write(f"#{description}\n")
                file.write(f"{setting} = {self.__dict__[setting]}\n\n")

class ExtensionSet(set):
    """converts a file extension to lowercase before checking if it exists"""
    def __contains__(self, item : str):
        return super().__contains__(item.lower())

class ThreadWrapper(threading.Thread): 
    """thread that supports return values with .join()
    and that raises exceptions on the main thread"""
    def __init__(self, target=None, name=None, args=(), kwargs={}):
        threading.Thread.__init__(self, None, target, name, args, kwargs)
        self._return = None
    def run(self):
        try:
            self._return = self._target(*self._args, **self._kwargs)
        except Exception as e:
            self._return = e
    def join(self, *args, **kwargs):
        threading.Thread.join(self, *args, **kwargs)
        if isinstance(self._return, Exception): 
            exception = self._return #make sure it raises the exception on only the first time join() is called
            self._return = None
            if exception is not None: raise exception
        return self._return