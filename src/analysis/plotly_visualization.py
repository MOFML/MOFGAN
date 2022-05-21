import json

import numpy as np
import torch

import project_config
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def main():
    print(project_config.local.root)
    with open(f"{project_config.local.root}/input") as f:
        data = json.load(f)
    t = torch.tensor(data)[4][0]
    print(t.shape)
    # x, y, z = torch.meshgrid(t, indexing='ijk')

    x1 = list(range(32))
    y1 = list(range(32))
    z1 = list(range(32))
    print(t.min(), t.max())

    x, y, z = np.meshgrid(x1, y1, z1)
    print(x.shape)

    values = t.flatten()

    fig = go.Figure(data=go.Volume(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value=values.flatten(),
        isomin=12,
        isomax=24,
        opacity=0.1,
        surface_count=16,
    ))
    fig.write_html("generated_mof.html")
    # fig.show()

    # x = t[:, :, 0]
    # y = t[:, :, 1]
    # z = t[:, :, 2]
    # print(x.shape)
    # # exit()
    # fig = go.Figure(data=go.Volume(
    #     x=x.flatten(),
    #     y=y.flatten(),
    #     z=z.flatten(),
    #     value=t.flatten(),
    #     # isomin=-0.1,
    #     # isomax=0.8,
    #     opacity=0.1,  # needs to be small to see through all surfaces
    #     # surface_count=21,  # needs to be a large number for good volume rendering
    # ))
    # fig.show()
    # fig = make_subplots(rows=1, cols=2,
    #                     specs=[[{'is_3d': True}, {'is_3d': True}]],
    #                     subplot_titles=['Color corresponds to z', 'Color corresponds to distance to origin'],
    #                     )
    #
    # fig.add_trace(go.Surface(x=x, y=y, z=z, colorbar_x=-0.07), 1, 1)
    # fig.add_trace(go.Surface(x=x, y=y, z=z, surfacecolor=x ** 2 + y ** 2 + z ** 2), 1, 2)
    # fig.update_layout(title_text="Ring cyclide")
    # fig.show()

    # volume viz and transparency.


if __name__ == '__main__':
    main()
