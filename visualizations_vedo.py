import pandas as pd
from vedo import *
import colorcet

def basic_statistics(original_df, climate, obstacle, orientation, heat_SP, cool_SP):
    """
    Reports the basic statistics for each diagram:
    - minimum value encountered (and for which depth and height)
    - maximum value encountered (and for which depth and height)
    - average slope over depth
    - average slope over height
    This is all reported on the console, without writing to a file
    """

    # Select the necessary subset---this is the starting position:
    df = original_df[(original_df.climate == climate) &
                     (original_df.obstacle == obstacle) &
                     (original_df.orientation == orientation) &
                     (original_df.heat_SP == heat_SP) &
                     (original_df.cool_SP == cool_SP)]
    df = df.sort_values(by=['height', 'depth'])

    # report the basic statistics for this smaller dataset
    print(
        f'Basic statistics for the case (clim,obst,orient,hsp,csp)=({climate},{obstacle},{orientation},{heat_SP},{cool_SP})')

    # minimum and maximum values
    minvals = df.min()
    minidxs = df.idxmin()
    maxvals = df.max()
    maxidxs = df.idxmax()
    print(f'Minimum heating load [kWh/m2]: {minvals["heat_load [kWh/m2]"]}')
    print(f'  attained for (depth,height)=({df["depth"][minidxs["heat_load [kWh/m2]"]]},',
          f'{df["height"][minidxs["heat_load [kWh/m2]"]]})')
    print(f'Maximum heating load [kWh/m2]: {maxvals["heat_load [kWh/m2]"]}')
    print(f'  attained for (depth,height)=({df["depth"][maxidxs["heat_load [kWh/m2]"]]},',
          f'{df["height"][maxidxs["heat_load [kWh/m2]"]]})')
    print(f'Minimum cooling load [kWh/m2]: {minvals["cool_load [kWh/m2]"]}')
    print(f'  attained for (depth,height)=({df["depth"][minidxs["cool_load [kWh/m2]"]]},',
          f'{df["height"][minidxs["cool_load [kWh/m2]"]]})')
    print(f'Maximum cooling load [kWh/m2]: {maxvals["cool_load [kWh/m2]"]}')
    print(f'  attained for (depth,height)=({df["depth"][maxidxs["cool_load [kWh/m2]"]]},',
          f'{df["height"][maxidxs["cool_load [kWh/m2]"]]})')
    print(f'Minimum lighting load [kWh/m2]: {minvals["light_load [kWh/m2]"]}')
    print(f'  attained for (depth,height)=({df["depth"][minidxs["light_load [kWh/m2]"]]},',
          f'{df["height"][minidxs["light_load [kWh/m2]"]]})')
    print(f'Maximum lighting load [kWh/m2]: {maxvals["light_load [kWh/m2]"]}')
    print(f'  attained for (depth,height)=({df["depth"][maxidxs["light_load [kWh/m2]"]]},',
          f'{df["height"][maxidxs["light_load [kWh/m2]"]]})')
    print(f'Minimum primary energy [kWh/m2]: {minvals["primary [kWh/m2]"]}')
    print(f'  attained for (depth,height)=({df["depth"][minidxs["primary [kWh/m2]"]]},',
          f'{df["height"][minidxs["primary [kWh/m2]"]]})')
    print(f'Maximum primary energy [kWh/m2]: {maxvals["primary [kWh/m2]"]}')
    print(f'  attained for (depth,height)=({df["depth"][maxidxs["primary [kWh/m2]"]]},',
          f'{df["height"][maxidxs["primary [kWh/m2]"]]})')

    # average slopes
    diffs_height = df[df.depth == 1.6].sum() - df[df.depth == 0.0].sum()
    print(f'Average slopes over depth:')
    print(f'   heating load: {diffs_height["heat_load [kWh/m2]"] / (1.6 * 25)}')
    print(f'   cooling load: {diffs_height["cool_load [kWh/m2]"] / (1.6 * 25)}')
    print(f'  lighting load: {diffs_height["light_load [kWh/m2]"] / (1.6 * 25)}')
    print(f' primary energy: {diffs_height["primary [kWh/m2]"] / (1.6 * 25)}')

    diffs_depth = df[df.height == 0.49].sum() - df[df.height == 0.01].sum()
    print(f'Average slopes over height:')
    print(f'   heating load: {diffs_depth["heat_load [kWh/m2]"] / (0.48 * 81)}')
    print(f'   cooling load: {diffs_depth["cool_load [kWh/m2]"] / (0.48 * 81)}')
    print(f'  lighting load: {diffs_depth["light_load [kWh/m2]"] / (0.48 * 81)}')
    print(f' primary energy: {diffs_depth["primary [kWh/m2]"] / (0.48 * 81)}')


def cooling_diagram(verts, faces, ZC, filename):
    mesh1 = Mesh([verts, faces])
    mesh1.pointdata['cooling load'] = ZC         # you must first associate numerical data to its points
    mesh1.pointdata.select('cooling load')       # and make them "active"
    mesh1.cmap(colorcet.CET_L7)     # ('terrain')

    isol1 = mesh1.isolines(n=14).color('w')
    isol1.lw(7)

    cam1 = dict(
        position=(3.85577, -1.73075, 1.59065),
        focal_point=(0.838932, 0.219305, 0.82), # 0.925060),
        viewup=(0,0,1),   # (-0.139573, 0.112697, 0.983778),
        distance=3.73636,
        clipping_range=(1.5, 6.11313),
    )
    light1 = Light(pos=(2,0,2), focal_point=cam1["focal_point"], c='w', intensity=1)

    plt1 = Plotter(N=1, size=(1400,1300),
                   axes=dict(xtitle='depth',
                            xtitle_offset=0.175,
                            xtitle_size=0.0175,
                            xlabel_size=0.012,
                            xaxis_rotation=90,
                            xygrid=True,
                            ytitle='height',
                            ytitle_offset=-0.115,
                            ytitle_size=0.0175,
                            ylabel_size=0.012,
                            yshift_along_x=1.0,
                            ylabel_offset=-2.5,
                            yaxis_rotation=90,
                            yzgrid=True,
                            ztitle='cooling load (kWh/m2)',
                            ztitle_offset=0.05,
                            ztitle_size=0.0175,
                            zlabel_size=0.012,
                            zaxis_rotation=45,
                            zrange=(0.5,1.5),
                            z_values_and_labels=[(i, f'{45*i-2.5:.2f}') for i in np.linspace(0.5,1.5,10)],
                            zxgrid2=True,
                            axes_linewidth=3,
                            grid_linewidth=2,
                            number_of_divisions=16,
                            text_scale=1.8)).parallel_projection(value=True)
    plt1.show(mesh1, isol1, light1, camera=cam1, interactive=False, zoom=1.25)
    plt1.screenshot(filename)
    plt1.close()


def heating_diagram(verts, faces, ZH, filename):
    mesh2 = Mesh([verts, faces])
    mesh2.pointdata['heating load'] = ZH
    mesh2.pointdata.select('heating load')
    mesh2.cmap(colorcet.CET_L3)    # ('terrain')

    isol2 = mesh2.isolines(n=27).color('w')
    isol2.lw(3)

    cam2 = dict(
        position=(-3, -2.25, 2.0),
        focal_point=(0.75, 0.25, 0.4),
        viewup=(0,0,1),
        distance=3.0,
        clipping_range=(1.0, 6.0),
    )
    light2 = Light(pos=(-1,1,3), focal_point=cam2["focal_point"], c='w', intensity=1)

    plt2 = Plotter(N=1, size=(1000,1200),
             axes = dict(xtitle='depth',
                xtitle_offset=0.2,
                xtitle_size=0.0175,
                xtitle_position=0.17,
                xlabel_size=0.012,
                xaxis_rotation=90,
                xygrid=True,
                ytitle='height',
                ytitle_offset=0,
                ytitle_size=0.0175,
                ytitle_rotation=(-90,0,90),
                ytitle_position=0.75,
                ylabel_size=0.012,
                ylabel_offset=-2.5,
                yzgrid=False,
                yzgrid2=True,
                ztitle='heating load (kWh/m2)',
                ztitle_offset=-0.075,
                ztitle_size=0.0175,
                ztitle_rotation=(90,0,15),
                zlabel_size=0.012,
                zaxis_rotation=-105,
                zshift_along_x = 1,
                zrange=(0.0, 1.0),
                z_values_and_labels=[(i, f'{30*i+30:.2f}') for i in np.linspace(1/12, 1.0, 12)],
                zxgrid2=True,
                axes_linewidth=3,
                grid_linewidth=2,
                number_of_divisions=16,
                text_scale=1.8)).parallel_projection(value=True)
    plt2.show(mesh2, isol2, light2, camera=cam2, interactive=False, zoom=1.4)
    plt2.screenshot(filename)
    plt2.close()


def lighting_diagram(verts, faces, ZL, filename):
    mesh3 = Mesh([verts, faces])
    mesh3.pointdata['lighting load'] = ZL
    mesh3.pointdata.select('lighting load')
    mesh3.cmap(colorcet.CET_L17)    # ('terrain')

    isol3 = mesh3.isolines(n=17).color('w')
    isol3.lw(3)

    cam3 = dict(
        position=(-0.25, -5, 12.3),
        focal_point=(0.725, -0.25, 11.2),
        viewup=(0,0,1),
        distance=5,
        clipping_range=(1.0, 7.0),
    )
    light3 = Light(pos=(-0.25, -5, 14), focal_point=cam3["focal_point"], c='w', intensity=1.25)

    plt3 = Plotter(N=1, size=(1500,1000),
                   axes=dict(xtitle='depth',
                             xtitle_offset=0.2,
                             xtitle_size=0.0175,
                             xtitle_position=0.185,
                             xlabel_size=0.012,
                             xaxis_rotation=90,
                             xygrid=True,
                             ytitle='height',
                             ytitle_offset=-0.03,
                             ytitle_size=0.0175,
                             ytitle_rotation=(0, 0, 90),
                             ytitle_position=1.25,
                             ylabel_size=0.012,
                             ylabel_offset=0.75,
                             ylabel_justify='center-right',
                             y_values_and_labels=[(i, f'{(0.5-i):.1f}') for i in np.linspace(0.1, 0.4, 4)],
                             yzgrid=False,
                             yzgrid2=True,
                             ztitle='lighting load (kWh/m2)',
                             ztitle_offset=-0.19,
                             ztitle_size=0.0175,
                             zlabel_size=0.012,
                             zlabel_offset=-0.75,
                             zlabel_justify='center-left',
                             zaxis_rotation=-33,
                             zshift_along_x=1,
                             zrange=(10.7, 11.7),
                             z_values_and_labels=[(i, f'{i:.1f}') for i in np.linspace(10.7, 11.7, 11)],
                             zxgrid=False,
                             zxgrid2=True,
                             axes_linewidth=3,
                             grid_linewidth=2,
                             number_of_divisions=17,
                             text_scale=1.8)).parallel_projection(value=True)
    plt3.show(mesh3, isol3, light3, camera=cam3, interactive=False, zoom=1.95)
    plt3.screenshot(filename)
    plt3.close()


def primary_diagram(verts, faces, ZT, filename):
    mesh4 = Mesh([verts, faces])
    mesh4.pointdata['primary energy'] = ZT         # you must first associate numerical data to its points
    mesh4.pointdata.select('primary energy')       # and make them "active"
    mesh4.cmap(colorcet.CET_L20)     # ('terrain')

    isol4 = mesh4.isolines(n=22).color('w')
    isol4.lw(3)

    cam3 = dict(
        position=(-4, -5, 3),
        focal_point=(0.4, -0.3, 0.65),
        viewup=(0,0,1),
        distance=5,
        clipping_range=(1.0, 7.0),
    )
    light3 = Light(pos=(0.8, 0, 4.0), focal_point=cam3["focal_point"], c='w', intensity=1)

    cam4 = dict(
        position=(3.85577, -1.73075, 1.1),
        focal_point=(0.838932, 0.219305, 0.4),
        viewup=(0,0,1),
        distance=3.73636,
        clipping_range=(1.0, 7.0),
    )
    light4 = Light(pos=(2,0,2), focal_point=cam4["focal_point"], c='w', intensity=1)

    plt4 = Plotter(N=1, size=(1200,1000),
                   axes=dict(xtitle='depth',
                            xtitle_offset=0.175,
                            xtitle_size=0.0175,
                            xtitle_position=0.16,
                            xlabel_size=0.012,
                            xaxis_rotation=90,
                            xygrid=True,
                            ytitle='height',
                            ytitle_offset=0.03,
                            ytitle_position=0.775,
                            ytitle_size=0.0175,
                            ytitle_rotation=(-90, 0, 90),
                            ylabel_size=0.012,
                            # yshift_along_x=1.0,
                            # ylabel_offset=0.7,
                            ylabel_justify='center-right',
                            yaxis_rotation=0,
                            yzgrid=False,
                            yzgrid2=True,
                            ztitle='primary energy (kWh/m2)',
                            ztitle_offset=-0.27,
                            ztitle_position=0.99,
                            ztitle_size=0.0175,
                            zlabel_size=0.012,
                            zlabel_justify='center-left',
                            zlabel_offset=-0.6,
                            zshift_along_x=1.0,
                            zaxis_rotation=-60,
                            zrange=(0.0,1.0),
                            z_values_and_labels=[(i, f'{190+100*i:.2f}') for i in np.linspace(1/16,1.0,16)],
                            zxgrid2=True,
                            axes_linewidth=3,
                            grid_linewidth=2,
                            number_of_divisions=16,
                            text_scale=1.8)).parallel_projection(value=True)
    plt4.show(mesh4, isol4, light3, camera=cam3, interactive=False, zoom=2.35)
    plt4.screenshot(filename)
    plt4.close()


def create_diagrams(original_df, climate, obstacle, orientation, heat_SP, cool_SP):
    # Select the necessary subset---this is the starting position:
    df = original_df[(original_df.climate==climate) &
                     (original_df.obstacle==obstacle) &
                     (original_df.orientation==orientation) &
                     (original_df.heat_SP==heat_SP) &
                     (original_df.cool_SP==cool_SP)]
    df = df.sort_values(by=['height', 'depth'])

    xsize = df['depth'].nunique()
    ysize = df['height'].nunique()

    X = df['depth'].to_numpy()
    Y = df['height'].to_numpy()
    Ybackwards = 0.5-Y         # reverse the Y-values in order to Y-mirror the lighting load diagram

    ZC = (df['cool_load [kWh/m2]'].to_numpy() + 2.5) / 45
    ZH = (df['heat_load [kWh/m2]'].to_numpy() - 30) / 30
    ZL = df['light_load [kWh/m2]'].to_numpy()
    ZT = (df['primary [kWh/m2]'].to_numpy() - 190 ) / 100

    # mesh vertices
    verts1 = list(zip(X,Y,ZC))
    verts2 = list(zip(X,Y,ZH))
    verts3 = list(zip(X,Ybackwards,ZL))
    verts4 = list(zip(X,Y,ZT))

    # mesh faces, i.e., triangles
    faces = [(xsize*j+i, xsize*j+i+1, xsize*j+xsize+i+1) for i in range(xsize-1) for j in range(ysize-1)]+\
            [(xsize*j+i, xsize*j+xsize+i, xsize*j+xsize+i+1) for i in range(xsize-1) for j in range(ysize-1)]

    # create each set of diagrams separately (because of different camera settings)
    # cooling_diagram(verts1, faces, ZC, filename=f'fig_cooling_cl{climate}_ob{obstacle}_or{orientation}_hs{heat_SP}_cs{cool_SP}.png')
    # heating_diagram(verts2, faces, ZH, filename=f'fig_heating_cl{climate}_ob{obstacle}_or{orientation}_hs{heat_SP}_cs{cool_SP}.png')
    # lighting_diagram(verts3, faces, ZL, filename=f'fig_lighting_cl{climate}_ob{obstacle}_or{orientation}_hs{heat_SP}_cs{cool_SP}.png')
    primary_diagram(verts4, faces, ZT, filename=f'fig_primary_cl{climate}_ob{obstacle}_or{orientation}_hs{heat_SP}_cs{cool_SP}.png')


if __name__=="__main__":
    # read the simulation results from the external csv file
    original_df = pd.read_csv("collected_results.csv")

    # add the column for the equivalent primary energy - already added!
    # original_df["primary [kWh/m2]"] = 1.092*original_df["heat_load [kWh/m2]"]/0.3 \
    #                                 + 3.317*original_df["cool_load [kWh/m2]"]/3.0 \
    #                                 + 3.317*original_df["light_load [kWh/m2]"]

    # The starting position is
    # New York (cl=4), no obstacles (0), south orientation (0),
    # heating (21) and cooling (24) set points as in the PNNL model,
    # and it is present in every diagram set (never mind doing the job several times :)
    basic_clim = 4
    basic_obst = 0
    basic_orient = 0
    basic_heat_SP = 21
    basic_cool_SP = 24

    # Each diagram set will need its own z-range and cameras' settings.
    # Since these diagrams will be created just once,
    # the necessary settings will be hard-coded this time,
    # based on the values found below.

    # Diagram set 1: Impact of climate
    # ds1_tuples = [(clim, basic_obst, basic_orient, basic_heat_SP, basic_cool_SP) for clim in range(6)]
    # for (clim, obst, orient, heat_SP, cool_SP) in ds1_tuples:
        # basic_statistics(original_df, clim, obst, orient, heat_SP, cool_SP)
        # create_diagrams(original_df, clim, obst, orient, heat_SP, cool_SP)

    # Diagram set 2: Impact of obstacles
    # ds2_tuples = [(basic_clim, obst, basic_orient, basic_heat_SP, basic_cool_SP) for obst in range(5)]
    # for (clim, obst, orient, heat_SP, cool_SP) in ds2_tuples:
    #    basic_statistics(original_df, clim, obst, orient, heat_SP, cool_SP)
    #    create_diagrams(original_df, clim, obst, orient, heat_SP, cool_SP)

    # Diagram set 3: Impact of orientation
    # ds3_tuples = [(basic_clim, basic_obst, orient, basic_heat_SP, basic_cool_SP) for orient in [0.0, 45.0, -45.0]]
    # for (clim, obst, orient, heat_SP, cool_SP) in ds3_tuples:
    #     basic_statistics(original_df, clim, obst, orient, heat_SP, cool_SP)
    #     create_diagrams(original_df, clim, obst, orient, heat_SP, cool_SP)

    # Diagram set 4: Impact of heating and cooling set points
    ds4_tuples = [(basic_clim, basic_obst, basic_orient, heat_SP, cool_SP) for heat_SP in [19, 21] for cool_SP in [24, 26]]
    for (clim, obst, orient, heat_SP, cool_SP) in ds4_tuples:
    #     basic_statistics(original_df, clim, obst, orient, heat_SP, cool_SP)
        create_diagrams(original_df, clim, obst, orient, heat_SP, cool_SP)
