from pysaic import PySaic, Stage
import ui.components
import pygame

#simple loading screen to keep the program responsive
#and handle ESC inputs
#any other stage will set PySaic.loading_thread
#and this stage monitors it 

class Loading(Stage):
    def start(self):
        self.ui = dict()
        self.ui["loading_overlay"] = ui.components.Overlay(PySaic.UI)
        self.ui["loading_text"] = ui.components.TextSurface(PySaic.UI, "Loading...", 30, [255]*3).place("midmid", "midmid")
        self.ui["loading_subtext"] = ui.components.TextSurface(PySaic.UI, "PRESS ESC TO CANCEL", 20, [255]*3, alpha=64).place("botmid", "botmid", (0, -50))
        PySaic.UI.add(self.ui)
        self.old_fps = PySaic.fps
        self.elapsed = 0
        PySaic.loading_thread.start()

    def update(self, events):        
        for event in events:
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                PySaic.stop_loading_flag = True 

        self.elapsed += 1
        if self.elapsed > PySaic.UI.settings.FADE_TIME*3:
            PySaic.change_fps(10)

        if not PySaic.loading_thread.is_alive():
            PySaic.loading_thread.join() #get any exceptions raised by the thread
            for value in self.ui.values():
                value.delete()
            del self.ui
            PySaic.change_fps(self.old_fps)
            PySaic.return_stage()