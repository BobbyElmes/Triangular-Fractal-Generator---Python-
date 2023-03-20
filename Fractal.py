import numpy as np
from PIL import Image
import math

# Plots a line between two coordinates

# Parameter list:
# from_coordinates  -  [y,x] point within the array of pixels to begin plotting the line
# to_coordinates    -  [y,x] point within the array to plot the line to
# thickness         -  how thick we want the line to be
# colour            -  [r,g,b] colour we want the line to have
# pixels            -  3d array representing the pixels e.g pixels[row][column][colour]

def plot_line(from_coordinates, to_coordinates, thickness, colour, pixels):

    # Figure out the boundaries of our pixel array
    max_x_coordinate = len(pixels[0])
    max_y_coordinate = len(pixels)

    # The distances along the x and y axis between the 2 points
    horizontal_distance = to_coordinates[1] - from_coordinates[1]
    vertical_distance = to_coordinates[0] - from_coordinates[0]

    # The total distance between the two points, used to calculate how far we want to step each time we look for a new pixel to colour in
    distance = math.sqrt((to_coordinates[1] - from_coordinates[1])**2 + (to_coordinates[0] - from_coordinates[0])**2)
    step = round(distance)

    # How far we will step forwards along the x and y axis each time we colour in a new pixel
    horizontal_step = horizontal_distance/step
    vertical_step = vertical_distance/step

    # At this point, we enter the loop to draw the line in our pixel array
    # Each iteration of the loop will add a new point along our line
    for i in range(step):
        
        # These 2 coordinates are the ones at the center of our line
        current_x_coordinate = round(from_coordinates[1] + (horizontal_step*i))
        current_y_coordinate = round(from_coordinates[0] + (vertical_step*i))
        
        if (current_x_coordinate > 0 and current_x_coordinate < max_x_coordinate and current_y_coordinate > 0 and current_y_coordinate < max_y_coordinate):
            pixels[current_y_coordinate][current_x_coordinate] = colour
            
        # Once we have the coordinates, we draw a 'point' (a square) around the coordinates of size 'thickness'
        for x in range (-thickness, thickness):
            for y in range (-thickness, thickness):
                x_value = current_x_coordinate + x
                y_value = current_y_coordinate + y

                if (x_value > 0 and x_value < max_x_coordinate and y_value > 0 and y_value < max_y_coordinate):
                    pixels[y_value][x_value] = colour


# Draws a triangle, then calls itself for each outer facing edge it has

# Parameter list:
# center            -  [y,x] point within the array of pixels where the center of the triangle lies
# side_length       -  length of each side of the triangle
# degrees_rotate    -  how many degrees we wish it to be rotated about the center
# thickness         -  how thick we want the edges to be
# colour            -  [r,g,b] colour we want each line to have
# pixels            -  3d array representing the pixels e.g pixels[row][column][colour]
# shrink_side_by    -  fraction we wish each side to be shrunk by with each iteration
# max_depth         -  maximum number of iterations 

def draw_triangle(center, side_length, degrees_rotate, thickness, colour, pixels, shrink_side_by, iteration, max_depth):
    
    # The height of an equilateral triangle is, h = ½(√3a) where 'a' is the side length
    triangle_height = side_length * math.sqrt(3)/2

    # The top corner
    top_corner = [center[0] - triangle_height/2, center[1]]

    # Bottom left corner
    bottom_left_corner = [center[0] + triangle_height/2, center[1] - side_length/2]

    # Bottom right corner
    bottom_right_corner = [center[0] + triangle_height/2, center[1] + side_length/2]

    if (degrees_rotate != 0):
        top_corner = rotate_coordinate_around_point(top_corner, center, degrees_rotate)
        bottom_left_corner = rotate_coordinate_around_point(bottom_left_corner, center, degrees_rotate)
        bottom_right_corner = rotate_coordinate_around_point(bottom_right_corner, center, degrees_rotate)

    lines = [[top_corner, bottom_left_corner], [top_corner, bottom_right_corner], [bottom_left_corner, bottom_right_corner]]
    line_number = 0

    # Draw a line between each corner to complete the triangle
    for line in lines:
        line_number += 1
        plot_line(line[0], line[1], thickness, colour, pixels)

        # Draw some new triangles
        if (iteration < max_depth and (iteration < 1 or line_number < 3)):
            gradient = (line[1][0] - line[0][0]) / (line[1][1] - line[0][1])

            new_side_length = side_length*shrink_side_by

            # Center of the line of the traingle we are drawing
            center_of_line = [(line[0][0] + line[1][0]) / 2, (line[0][1] + line[1][1]) / 2]

            new_center = []
            new_rotation = degrees_rotate

            # Amount we need to rotate the traingle by
            if (line_number == 1):
                new_rotation += 60
            elif (line_number == 2):
                new_rotation -= 60
            else:
                new_rotation += 180
            
            # In an ideal world this would be gradient == 0, but due to floating point division
            # we cannot ensure that this will always be the case
            if (gradient < 0.0001 and gradient > -0.0001):
                if (center_of_line[0] - center[0] > 0):
                    new_center = [center_of_line[0] + triangle_height * (shrink_side_by/2), center_of_line[1]]
                else:
                    new_center = [center_of_line[0] - triangle_height * (shrink_side_by/2), center_of_line[1]]
                    
            elif gradient != 0:
                
                # Calculate the normal to the gradient of the line we're going to draw a new triangle on
                difference_from_center = -1/gradient

                # Calculate the distance from the center of the line that the center of our new traingle will be
                distance_from_center = triangle_height * (shrink_side_by/2)

                # Calculate the size of the x axis, from the center of our line to the center of our new triangle
                x_length = math.sqrt((distance_from_center**2)/(1 + difference_from_center**2))

                # Figure out which way around the x direction needs to go
                if (center_of_line[1] < center[1] and x_length > 0):
                    x_length *= -1

                # Now calculate the y length and direction of the axis
                y_length = x_length * difference_from_center

                # Offset the center of the line with our new x and y values
                new_center = [center_of_line[0] + y_length, center_of_line[1] + x_length]

            draw_triangle(new_center, new_side_length, new_rotation, thickness, colour, pixels, shrink_side_by, iteration+1, max_depth)


# Rotates 'coordinate' around the 'center_point'

# Paramater list:
# coordinate    -  [y,x] point to rotate
# center_point  -  [y,x] point to rotate around
# degrees       -  degrees to rotate by

def rotate_coordinate_around_point(coordinate, center_point, degrees):
    # Subtract the point we are rotating around from our coordinate to remove the offset from 0,0
    x = (coordinate[0] - center_point[0])
    y = (coordinate[1] - center_point[1])

    # Python's cos and sin functions take radians instead of degrees
    radians = math.radians(degrees)

    # Calculate our rotated points 
    new_x = (x * math.cos(radians)) - (y * math.sin(radians))
    new_y = (y * math.cos(radians)) + (x * math.sin(radians))

    # Add back our offset we subtracted at the beginning to our rotated points and return
    return [new_x + center_point[0], new_y + center_point[1]]
    
# Define the size of our image
pixels = np.zeros( (10000,10000,3), dtype=np.uint8 )

# Draw 4 fractals
draw_triangle([3000,3000], 2000, 0, 0, [255,200,0], pixels, 1/2, 0, 9)
draw_triangle([3000,7000], 1200, 0, 0, [165, 242, 243], pixels, 2/3, 0, 9)
draw_triangle([7000,3000], 1000, 0, 0, [124,252,0], pixels, 2.28/3, 0, 9)
draw_triangle([7000,7000], 800, 0, 0, [203, 195, 227], pixels, 2.5/3, 0, 9)


# Turn our pixel array into a real picture
img = Image.fromarray(pixels)

# Show our picture, and save it
img.show()
img.save('Fractal.png')
