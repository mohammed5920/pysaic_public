PySaic is a demo using PyGame & Numba to render images & video mosaics in real-time with significant (1000x+) levels of detail.

![Demo](assets/readme/hybrid.gif "Preview")

It supports generating mosaics from both images and videos, using any combination of the two.

**Image** -> creates an image where each pixel is an image or a still frame from a video\
**Video** -> streams in a video and replaces each pixel with an image or a still frame from a video in real-time\
**Hybrid** -> creates an image where each pixel is a colour-matched clip from a video (RAM-intensive)

![Demo](assets/readme/video.gif "Preview")

## Performance
On a Ryzen 5 5600, using the entirety of [Parks & Recreation](https://en.wikipedia.org/wiki/Parks_and_Recreation) as the tileset:
- Image mosaics are generated instantly
- 1080p image/hybrid mosaics are rendered at 60-100fps (depending on zoom level) 
- 1080p videos are converted to mosaics at roughly 40fps
- RAM usage scales from around 100MB for the UI + rendering a 1080p image mosaic -> 1GB to stream 1080p video into a video mosaic -> around 8GB to comfortably zoom in and view a 1080p hybrid mosaic (lower is possible but increases CPU load)

## Analysis
Unlike other mosaic generators that work with only images, this script prioritises video support - when videos are used as tiles, the video is scanned end-to-end and analysed to extract distinct cuts (shots, scenery changes, etc). Videos are then stored as a collection of distinct key frames + how long the shot remains at roughly the same colour for. When videos are used as source material, the source video is streamed and analysed ahead of time by n seconds, allowing for real-time playback at a fixed RAM budget.

Performance in the analysis pipeline has also been prioritised. All media analysis happens at a fixed resolution using custom parallelised Numba-accelerated routines, allowing for images and videos to be analysed and colour-matched quite literally as fast as Pillow/CV2 can open them. RAM usage is kept to a minimum by keeping only 64x64 versions of all tiles and streaming the rest as needed.

## Rendering & Streaming
The renderer itself bypasses PyGame and renders all detail using parallelised Numba functions straight to NumPy arrays, which are then copied to the buffer surface and scaled. This allows for 100x faster rendering than in raw PyGame + Python alone.

Locking is used liberally here to avoid corrupting Numba's thread-unsafe data structures while rendering from multiple threads, allowing tiles to be streamed in while rendering safely.

So how does it store whole videos in RAM for each pixel at 1000x detail? It doesn't. PySaic implements a predictive layered streaming system modelled after video game tech designed to balance performance, I/O and RAM usage considerations. It can scale down to tens of megabytes for normal usage and maintains full video tile streaming comfortably with around 8GB of RAM budget. Zoom in on a complex hybrid mosaic, and the system will evict least recently used tiles to make space for a full resolution version of the frames requested that are then mip-mapped to fit on screen. This allows for 1 fetch for multiple levels of zoom.

## Limitations
- Python, PyGame & Pillow/OpenCV worked for the original idea I was writing at the time (render a low-res image mosaic to a file) but at a parallelised scale they're not the optimal choice of technology. A rewrite that directly uses FFMPEG & OpenGL would be significantly faster. 

- To save on RAM & CPU, nearest-neighbour downscaling is used in the tile streaming pipeline, which leads to aliasing when zooming between mip levels.

- The UI uses an alpha version of what eventually became [MiniUI](https://github.com/mohammed5920/miniui_public). I'm currently working on MiniUI 2, a declarative, primitive-based composable re-write of the engine that will simplify the UI DX immensely, and allow for complex layouts to be layered on top of the renderer for settings/saving UI.