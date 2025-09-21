from pysaic import PySaic, Stage
import util.misc
import os
import pygame
from tkinter.filedialog import askopenfilename, askdirectory
from tqdm import tqdm
from typing import Literal

import ui.components
import ui.util.exceptions
import ui.transform

UII = PySaic.UI
#home screen (choose mosaic mode, open tile folder & source media)
#all metadata is stored in a global metadata object called PySaic.temp
#once "go" is clicked and assuming all metadata is valid, it transfers to stage_mosaic

class Main(Stage):
    def start(self):
        self.ui_modes = ui.components.TB_Layout(UII).add_components({
                    "pic" : ui.components.PaneButton(UII, "assets/main/photo.jpg", (325, 75), "PHOTO", 24, 
                                                     click_func=self.clickon_mode, click_args=("pic",)),
                    "vid" : ui.components.PaneButton(UII, "assets/main/video.mp4", (325, 75), "VIDEO", 24, 
                                                     click_func=self.clickon_mode, click_args=("vid",)),
                    "hyb" : ui.components.PaneButton(UII, "assets/main/hybrid.mp4", (325, 75), "HYBRID", 24,
                                                     click_func=self.clickon_mode, click_args=("hyb",))
                })
        
        self.ui_tiles = ui.components.TB_Layout(UII).add_components({
                    "folder" : ui.components.TileButton(UII, "assets/tiles/folder.png", (325, 75), "Scan tiles", 24,
                                                      click_func=self.clickon_folder, toggled=True),
                    "pics" : ui.components.TileButton(UII, "assets/tiles/pictures.png", (325, 75), "Enable pictures", 24,
                                                      click_func=self.clickon_picbasic, toggled=PySaic.temp.enable_pics),
                    "vids" : ui.components.TileButton(UII, "assets/tiles/videos.png", (325, 75), "Enable videos", 24,
                                                      click_func=self.clickon_vidbasic, toggled=PySaic.temp.enable_vids)
                })
        
        self.ui_source = ui.components.TB_Layout(UII).add_components({
                    "go" : ui.components.TileButton(UII, "assets/tiles/check.png", (325, 75), "Go!", 24,
                                                        click_func=self.clickon_go),
                    "choose" : ui.components.TileButton(UII, "assets/tiles/source.png", (325, 75), "Choose media", 24,
                                                        click_func=self.clickon_media)
                })
        
        self.root = ui.components.LR_Layout(UII, hz_padding=20).place("midmid", "midmid").add_components({
                "mode" : self.ui_modes,
                "tiles" : self.ui_tiles,
                "last" : self.ui_source,
            })        
        
        UII.add({"main_overlay" : ui.components.Overlay(UII).place("midmid", "midmid"), "main_root" : self.root})
        UII.add({"test" : ui.components.TextSurface(UII, "test", 36, [255]*3).place("midmid", "topleft", (800, 100))})
        self.change_mode(PySaic.temp.mode or "pic")

    def change_mode(self, new_mode : Literal["hyb", "pic", "vid"]):
        """refreshes the home screen when the user selects one of the three modes"""
        for mode in ("pic", "vid", "hyb"):
            self.ui_modes[mode].toggle(mode == new_mode) #visually toggle the buttons
        PySaic.temp.mode = new_mode
        
        #middle (choose tile folder / enable pics/vids)
        if new_mode != "hyb": 
            if "vids" not in self.ui_tiles: 
                self.ui_tiles["vids"] = ui.components.TileButton(UII, "assets/tiles/videos.png", (325, 75), "Enable videos", 24,
                                                      click_func=self.clickon_vidbasic, toggled=PySaic.temp.vid_tiles and PySaic.temp.enable_vids)
        elif "vids" in self.ui_tiles: #delete element if hybrid mode is enabled (vids are always enabled in this mode)
            self.ui_tiles.queue_del_component("vids")
        
        #right - when to ask for media location - no media, invalid vid, invalid photo
        ext = PySaic.temp.source_path.split(".")[-1] if PySaic.temp.source_path else None
        if "choose" not in self.ui_source:
            if not PySaic.temp.source_path \
            or (new_mode == "vid" and ext not in PySaic.vid_extensions) \
            or (new_mode != "vid" and ext not in PySaic.pic_extensions):
                if PySaic.temp.source_path in self.ui_source: self.ui_source.queue_del_component(PySaic.temp.source_path)
                self.ui_source.add_components({"choose" : ui.components.TileButton(UII, "assets/tiles/source.png", (325, 75), "Choose media", 24,
                                                        click_func=self.clickon_media)}) 
        
        #when to respawn the chosen media preview - switch from mode where media was invalid to mode where media is valid
        if PySaic.temp.source_path not in self.ui_source and PySaic.temp.source_path:     
            if (new_mode == "vid" and ext in PySaic.vid_extensions) \
            or (new_mode != "vid" and ext in PySaic.pic_extensions):
                if "choose" in self.ui_source: self.ui_source.queue_del_component("choose")
                self.ui_source.add_components({PySaic.temp.source_path : ui.components.PaneButton(UII, PySaic.temp.source_path, (325, 75), "CHANGE MEDIA", 24,
                                                                    click_func=self.clickon_media)}) 

        self.refresh_mode()

    def refresh_mode(self):
        """refresh any UI elements when state changes"""
        self.ui_tiles["folder"]["text"].change_text("Scan tiles" if not PySaic.temp.tile_folder else 
                                           f"{'...' if len(PySaic.temp.tile_folder) > 12 else ''}{PySaic.temp.tile_folder[-12:]}")
        self.check_go()

    def check_go(self):
        """validates the selected media for the given mosaic mode"""
        if (PySaic.temp.source_path in self.ui_source and PySaic.temp.source_path) and \
           ((PySaic.temp.mode == "hyb" and PySaic.temp.vid_tiles) \
        or (PySaic.temp.mode != "hyb" and (self.ui_tiles["pics"].is_toggled or self.ui_tiles["vids"].is_toggled))):
            self.ui_source["go"].toggle(True)
        else:
            self.ui_source["go"].toggle(False)
    
    def clickon_mode(self, mode):
        if PySaic.temp.mode == mode:
            return
        self.change_mode(mode)

    def clickon_folder(self):
        """user clicks on 'scan tiles'\n
        recursively scans a tile folder and collects all the video/picture file names"""
        if not (folder := askdirectory(mustexist=True, title="Choose a folder with tiles...")):
            return
        
        def scan_folder(parent_folder):
            pic_tiles = []
            vid_tiles = []
            print("Scanning folder...")
            for folder, _, files in tqdm((os.walk(parent_folder))):
                for file in files:
                    if PySaic.stop_loading_flag:
                        util.misc.log("User cancelled folder scan.")
                        return
                    if file.split(".")[-1] in PySaic.pic_extensions:
                        pic_tiles.append(f"{folder}\\{file}")
                    elif file.split(".")[-1] in PySaic.vid_extensions:
                        vid_tiles.append(f"{folder}\\{file}")
            return(parent_folder, pic_tiles, vid_tiles)
        
        def returnfrom_scanfolder(self):
            if not PySaic.stop_loading_flag:
                PySaic.temp.tile_folder, PySaic.temp.pic_tiles, PySaic.temp.vid_tiles = PySaic.loading_thread.join()
                self.ui_tiles["pics"].toggle(PySaic.temp.pic_tiles)
                if "vids" in self.ui_tiles: self.ui_tiles["vids"].toggle(PySaic.temp.vid_tiles)
                self.refresh_mode()
            else:  #folder scan was halted by user, don't change any tile info
                PySaic.stop_loading_flag = False 
            PySaic.loading_thread = None    
        
        #switch to the loading stage because os.walk can take a long time and freeze the program
        PySaic.loading_thread = util.misc.ThreadWrapper(target=scan_folder, args=(folder,)) 
        PySaic.transfer_stage("loading", return_func=returnfrom_scanfolder)

    def clickon_picbasic(self):
        """user clicks on 'enable pictures'"""
        if not PySaic.temp.tile_folder:
            UII.toast("Choose a folder first!")
            return    
        if not PySaic.temp.pic_tiles:
            UII.toast("No photos found in the folder!")
            return
        self.ui_tiles["pics"].toggle()
        self.check_go()
    
    def clickon_vidbasic(self):
        """user clicks on 'enable videos'"""
        if not PySaic.temp.tile_folder:
            UII.toast("Choose a folder first!")
            return    
        if not PySaic.temp.vid_tiles:
            UII.toast("No videos found in the folder!")
            return
        self.ui_tiles["vids"].toggle()
        self.check_go()

    def clickon_media(self):
        """user clicks on 'change media' if media has been loaded in / 'choose media' if it hasn't (both lead here)"""
        if not (media := askopenfilename()) or media == PySaic.temp.source_path:
            return
        ext = media.split(".")[-1]
        if (PySaic.temp.mode == "vid" and ext not in PySaic.vid_extensions) or \
           (PySaic.temp.mode != "vid" and ext not in PySaic.pic_extensions):
            UII.toast(f"Not a valid {'video' if PySaic.temp.mode == 'vid' else 'picture'}!")
            return
        
        try:
            media_test = ui.components.PaneButton(UII, media, (325, 75), "CHANGE MEDIA", 24, click_func=self.clickon_media)
            if "choose" in self.ui_source: self.ui_source.queue_del_component("choose")
            else: self.ui_source.queue_del_component(PySaic.temp.source_path)
            PySaic.temp.source_path = media
            self.ui_source.add_components({PySaic.temp.source_path : media_test})
        except ui.util.exceptions.UIMediaException as e:
            UII.toast(f"{e} while opening media!")
            
        self.check_go()

    def clickon_go(self):
        """user clicks on 'go'
        \n performs the necessary validation to check that the program can proceed"""
        ext  = PySaic.temp.source_path.split(".")[-1] if PySaic.temp.source_path else None
        if not PySaic.temp.tile_folder:
            UII.toast("Please choose a valid tile folder!")
            assert not self.ui_source["go"].is_toggled
            return
        if (PySaic.temp.mode == "hyb" and not PySaic.temp.vid_tiles) \
        or (PySaic.temp.mode != "hyb" and not (PySaic.temp.vid_tiles or PySaic.temp.pic_tiles)):
            UII.toast(f"Please select a folder with {'media' if PySaic.temp.mode != 'hyb' else 'videos'}!")
            assert not self.ui_source["go"].is_toggled
            return
        if not PySaic.temp.source_path \
        or (ext not in PySaic.vid_extensions and PySaic.temp.mode == "vid") \
        or (ext not in PySaic.pic_extensions and PySaic.temp.mode != "vid"):
            UII.toast(f"Please choose a valid {'video' if PySaic.temp.mode == 'vid' else 'picture'}!")
            assert not self.ui_source["go"].is_toggled
            return
        if PySaic.temp.mode != "hyb" and not (self.ui_tiles["pics"].is_toggled or self.ui_tiles["vids"].is_toggled):
            UII.toast(f"Please enable at least one type of tile!")
            assert not self.ui_source["go"].is_toggled
            return
        
        assert self.ui_source["go"].is_toggled
        PySaic.temp.enable_pics = self.ui_tiles["pics"].is_toggled
        PySaic.temp.enable_vids = PySaic.temp.mode == "hyb" or self.ui_tiles["vids"].is_toggled #force videos on for hybrid mode
        if PySaic.mosaic != PySaic.temp or not PySaic.return_stage():
            PySaic.transfer_stage("mosaic")
        
    def update(self, events):
        for event in events:
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                PySaic.return_stage()

    def cleanup(self):
        UII.root.queue_del_component("main_overlay")
        UII.root.queue_del_component("main_root")
        del self.ui_modes
        del self.ui_source
        del self.ui_tiles
        del self.root