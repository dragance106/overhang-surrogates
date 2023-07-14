from eppy import modeleditor
from eppy.modeleditor import IDF

import math


def create_m1_model(climate, obstacle, height, depth, orientation=0.0, heat_SP=21.0, cool_SP=24.0):
    """
    This method creates an office cell model with the attached overhang
    with <depth> m, that is positioned at <height> m above the window.
    As the office cell model is assumed to be a part of a lengthy building,
    whose window will extend the length of the building and will be shaded by the corresponding overhang,
    in the present office cell model the attached overhang is sufficiently extended to the left and the right of the window,
    to include parts of the overall overhang that influence the window of the office cell
    (one length of the office cell to the left and to the right).

    Besides the <height> and the <depth>,
    the method takes as parameters also the index <climate> of the location/climate file:
    0 = Dubai (zone 0B, extremely hot, dry)
    1 = Honolulu (zone 1A, very hot, humid)
    2 = Tucson (zone 2B, hot, dry)
    3 = San Diego (zone 3A, warm, marine)
    4 = New York (zone 4A, mixed, humid)
    5 = Denver (zone 5B, cool, dry)

    as well as the presence of <obstacle> on the southern side of the office cell:
    0 = no obstacles
    1 = medium obstacle from 11am to 2pm
    2 = high obstacle from 11am to 2pm
    3 = medium obstacle from 3pm to 5pm
    4 = medium obstacle from 8am to 10am and medium obstacle from 3pm to 5pm

    The office cell is rotated for <orientation> degrees away from true south
    (clockwise is positive; the obstacles are not rotated).

    Finally, the heating and the cooling setpoints are set at <heat_SP> and <cool_SP>, respectively.

    IMPORTANT NOTE:
    EnergyPlus will report errors if any two vertices are placed less than 0.01m apart.
    Height is thus set to at least 0.01m by taking min(height, 0.01).
    Depth, if less than 0.01, will skip generating the overhang.

    :return: The idf object ready to be simulated.
    """

    idd_file = '../Energy+.idd'
    try:
        IDF.setiddname(idd_file)
    except modeleditor.IDDAlreadySetError as e:
        pass

    # load the starting building model
    idf_name = ["m1Dubai.idf", "m1Honolulu.idf", "m1Tucson.idf", "m1SanDiego.idf", "m1NewYork.idf", "m1Denver.idf"]
    epw_name = ["m1Dubai.epw", "m1Honolulu.epw", "m1Tucson.epw", "m1SanDiego.epw", "m1NewYork.epw", "m1Denver.epw"]

    idf = IDF(idf_name[climate], epw_name[climate])

    # set the orientation
    building = idf.idfobjects['BUILDING'][0]
    building.North_Axis = orientation

    # the cooling setpoint should be at least two degrees Celsius higher than the heating setpoint
    # otherwise, correct it as you see fit
    if cool_SP - heat_SP < 2.0:
        mid = (heat_SP + cool_SP)/2
        heat_SP = mid - 1.5
        cool_SP = mid + 1.5

    # set a different heating schedule, if necessary
    schedules = idf.idfobjects['SCHEDULE:COMPACT']
    if heat_SP != 21.0:
        heat_sch = [s for s in schedules if s.Name == 'HTGSETP_SCH_YES_OPTIMUM'][0]
        heat_sch.Field_4 = heat_SP - 5.4    # 15.6
        heat_sch.Field_6 = heat_SP - 3.4    # 17.6
        heat_sch.Field_8 = heat_SP - 1.4    # 19.6
        heat_sch.Field_10 = heat_SP         # 21.0
        heat_sch.Field_12 = heat_SP - 5.4   # 15.6
        heat_sch.Field_15 = heat_SP - 5.4   # 15.6
        heat_sch.Field_17 = heat_SP - 3.2   # 17.8
        heat_sch.Field_19 = heat_SP - 1.0   # 20.0
        heat_sch.Field_21 = heat_SP         # 21.0
        heat_sch.Field_23 = heat_SP - 5.4   # 15.6
        heat_sch.Field_26 = heat_SP - 5.4   # 15.6
        heat_sch.Field_29 = heat_SP - 5.4   # 15.6
        heat_sch.Field_31 = heat_SP - 3.2   # 17.8
        heat_sch.Field_33 = heat_SP - 1.0   # 20.0
        heat_sch.Field_35 = heat_SP         # 21.0
        heat_sch.Field_37 = heat_SP - 5.4   # 15.6
        heat_sch.Field_40 = heat_SP - 5.4   # 15.6

    # set a different cooling schedule, if necessary
    if cool_SP != 24.0:
        cool_sch = [s for s in schedules if s.Name == 'CLGSETP_SCH_YES_OPTIMUM'][0]
        cool_sch.Field_4 = cool_SP + 2.7    # 26.7
        cool_sch.Field_6 = cool_SP + 1.7    # 25.7
        cool_sch.Field_8 = cool_SP + 1.0    # 25.0
        cool_sch.Field_10 = cool_SP         # 24.0
        cool_sch.Field_12 = cool_SP + 2.7   # 26.7
        cool_sch.Field_15 = cool_SP + 2.7   # 26.7
        cool_sch.Field_17 = cool_SP + 1.6   # 25.6
        cool_sch.Field_19 = cool_SP + 1.0   # 25.0
        cool_sch.Field_21 = cool_SP         # 24.0
        cool_sch.Field_23 = cool_SP + 2.7   # 26.7
        cool_sch.Field_26 = cool_SP + 2.7   # 26.7
        cool_sch.Field_28 = cool_SP + 1.6   # 25.6
        cool_sch.Field_30 = cool_SP + 1.0   # 25.0
        cool_sch.Field_32 = cool_SP         # 24.0
        cool_sch.Field_34 = cool_SP + 2.7   # 26.7
        cool_sch.Field_37 = cool_SP + 2.7   # 26.7
        cool_sch.Field_40 = cool_SP + 2.7   # 26.7

    # assuming that we are interested in shading the first window only...
    window = idf.idfobjects['FENESTRATIONSURFACE:DETAILED'][0]
    zmax = max(window.Vertex_1_Zcoordinate, window.Vertex_2_Zcoordinate,
               window.Vertex_3_Zcoordinate, window.Vertex_4_Zcoordinate)
    xmin = min(window.Vertex_1_Xcoordinate, window.Vertex_2_Xcoordinate,
                window.Vertex_3_Xcoordinate, window.Vertex_4_Xcoordinate)
    xmax = max(window.Vertex_1_Xcoordinate, window.Vertex_2_Xcoordinate,
                window.Vertex_3_Xcoordinate, window.Vertex_4_Xcoordinate)
    width = xmax - xmin

    # attach an overhang to the "Office_Cell_Wall_South" wall if depth is at least 0.01m
    if depth>=0.01:
        # height should be at least 0.01m to avoid EnergyPlus reporting errors about vertex coincidences
        height = max(height, 0.01)

        overhang = idf.newidfobject('SHADING:ZONE:DETAILED')

        overhang.Name = 'Overhang'
        overhang.Base_Surface_Name = 'Office_Cell_Wall_South'
        overhang.Transmittance_Schedule_Name = ''
        overhang.Number_of_Vertices = 4

        overhang.Vertex_1_Xcoordinate = xmin - width
        overhang.Vertex_2_Xcoordinate = xmin - width
        overhang.Vertex_3_Xcoordinate = xmax + width
        overhang.Vertex_4_Xcoordinate = xmax + width

        overhang.Vertex_1_Ycoordinate = 0
        overhang.Vertex_2_Ycoordinate = -depth
        overhang.Vertex_3_Ycoordinate = -depth
        overhang.Vertex_4_Ycoordinate = 0

        overhang.Vertex_1_Zcoordinate = zmax + height
        overhang.Vertex_2_Zcoordinate = zmax + height
        overhang.Vertex_3_Zcoordinate = zmax + height
        overhang.Vertex_4_Zcoordinate = zmax + height

        reflectance = idf.newidfobject('SHADINGPROPERTY:REFLECTANCE')
        reflectance.Shading_Surface_Name = overhang.Name
        reflectance.Diffuse_Solar_Reflectance_of_Unglazed_Part_of_Shading_Surface = 0.4
        reflectance.Diffuse_Visible_Reflectance_of_Unglazed_Part_of_Shading_Surface = 0.4
        reflectance.Fraction_of_Shading_Surface_That_Is_Glazed = 0
        reflectance.Glazing_Construction_Name = ''

    # add obstacles:
    # 0 = no obstacles
    # 1 = left=15, right=30, up=30
    # 2 = left=15, right=30, up=60
    # 3 = left=-45, right=75, up=30
    # 4 = left=60, right=-30, up=30 and left=-45, right=75, up=30
    if obstacle>0:
        # first we need the center of the window
        xc = (window.Vertex_1_Xcoordinate+window.Vertex_2_Xcoordinate
              +window.Vertex_3_Xcoordinate+window.Vertex_4_Xcoordinate)/4
        yc = (window.Vertex_1_Ycoordinate+window.Vertex_2_Ycoordinate
              +window.Vertex_3_Ycoordinate+window.Vertex_4_Ycoordinate)/4
        zc = (window.Vertex_1_Zcoordinate+window.Vertex_2_Zcoordinate
              +window.Vertex_3_Zcoordinate+window.Vertex_4_Zcoordinate)/4

        if obstacle==1:
            create_obstacle(idf, xc, yc, zc, 20.0, 15.0, 30.0, 30.0)
        elif obstacle==2:
            create_obstacle(idf, xc, yc, zc, 20.0, 15.0, 30.0, 60.0)
        elif obstacle==3:
            create_obstacle(idf, xc, yc, zc, 20.0, -45.0, 75.0, 30.0)
        elif obstacle==4:
            create_obstacle(idf, xc, yc, zc, 20.0, 60.0, -30.0, 30.0)
            create_obstacle(idf, xc, yc, zc, 20.0, -45.0, 75.0, 30.0)

    # idf.saveas(f'm1-climate={climate}-obstacle={obstacle}-height={height}-depth={depth}.idf')
    # idf.run() will automatically save a copy of this file with a random suffix on the disk,
    # run energyplus on the copy, and then delete that copy
    # (but probably leave other files that energyplus has generated while running)
    # hence just prepare the idfname, and also
    # make sure that each idf has a distinct filename
    idf.idfname = f'm1_cl_{climate}_ob_{obstacle}_h_{height}_d_{depth}_or_{orientation}_hsp_{heat_SP}_csp_{cool_SP}.idf'

    return idf


def create_obstacle(idf, x, y, z, dist, left, right, up):
    """
    The method creates a southern obstacle - a single Shading:Site:Detailed object - that
    obstruct the view towards south, i.e., prevents the sunshine from coming through,
    from the reference point <(x,y,z)>
    to <left> degrees on the left,
    to <right> degrees on the right, and
    to <up> degrees upwards.
    The obstacle is at distance <dist> towards south,
    while the bottom of the obstacle is always on the ground.
    The obstacle is then added to the existing <idf> object.
    """

    xm = x - dist * math.tan(math.radians(right))
    xp = x + dist * math.tan(math.radians(left))

    zp = z + dist * math.tan(math.radians(up))

    obstacle = idf.newidfobject('SHADING:SITE:DETAILED')
    obstacle.Name = f'obstacle_{left}_{right}'
    obstacle.Transmittance_Schedule_Name = ''
    obstacle.Number_of_Vertices = 4

    obstacle.Vertex_1_Xcoordinate = xm
    obstacle.Vertex_2_Xcoordinate = xm
    obstacle.Vertex_3_Xcoordinate = xp
    obstacle.Vertex_4_Xcoordinate = xp

    obstacle.Vertex_1_Ycoordinate = y-dist
    obstacle.Vertex_2_Ycoordinate = y-dist
    obstacle.Vertex_3_Ycoordinate = y-dist
    obstacle.Vertex_4_Ycoordinate = y-dist

    obstacle.Vertex_1_Zcoordinate = zp
    obstacle.Vertex_2_Zcoordinate = 0
    obstacle.Vertex_3_Zcoordinate = 0
    obstacle.Vertex_4_Zcoordinate = zp

    reflectance = idf.newidfobject('SHADINGPROPERTY:REFLECTANCE')
    reflectance.Shading_Surface_Name = obstacle.Name
    reflectance.Diffuse_Solar_Reflectance_of_Unglazed_Part_of_Shading_Surface = 0.4
    reflectance.Diffuse_Visible_Reflectance_of_Unglazed_Part_of_Shading_Surface = 0.4
    reflectance.Fraction_of_Shading_Surface_That_Is_Glazed = 0
    reflectance.Glazing_Construction_Name = ''
