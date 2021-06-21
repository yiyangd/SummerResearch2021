point = []
intrin = {"model": 0}

def rs2_project_point_to_pixel(pixel[2], intrin, point[3]):
    x = point[0] / point[2]
    y = point[1] / point[2]
    if intrin.model == 0:
        r2 = x * x + y * y
        