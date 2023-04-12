# Scroll Viewer

![screenshot](./cover.png)

This is a web-based scroll viewer for use in the [Vesuvius Challenge](https://scrollprize.org/). Currently supports viewing the campfire scroll with fast, webgl-based rendering, color themes, and layer isolation features. 

Contributions are most welcome! Some future plans for the project:
- View larger scrolls (ie the fragments) via progressive loading and level of detail
- View ground-truth segmentations
- Annotate scroll layers and virtually unwrap (this can be done really fast via webgl magic)
- Integrate with your custom ink detection backeds (ie ink-id)

## Usage Instructions
1. Clone this repository
2. Due to size restrictions, the campfire scroll data isn't included in this repository. Download the campfire.zip from [here](https://scrollprize.org/data)
3. Unzip campfire.zip and place it in the home directory of this repository
4. Run the conversion script with `python converttoraw.py`. This converts the campfire scroll data into an 8-bit 3d texture file, which can be loaded into the scroll viewer.
5. Run the scroll viewer with any http server. I recommend `python -m http.server 8000`, which can be accessed at `localhost:8000` in your web browser.
6. If you have any issues at all, don't hesistate to open an issue ticket, reach out on the [Vesuvius Challenge Discord](https://discord.gg/6FgWYNjb4N) or [on twitter](https://twitter.com/LukeFarritor).
