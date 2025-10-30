def is_right_triangle(xA, yA, xB, yB, xC, yC):
    # Calculate the squared lengths of the sides of the triangle
    AB_squared = (xB - xA)**2 + (yB - yA)**2
    BC_squared = (xC - xB)**2 + (yC - yB)**2
    CA_squared = (xA - xC)**2 + (yA - yC)**2
    
    # Sort the squared lengths
    sides_squared = sorted([AB_squared, BC_squared, CA_squared])
    
    # Check the Pythagorean theorem for the sorted lengths
    return sides_squared[0] + sides_squared[1] == sides_squared[2]

# Read the input from stdin
xA, yA = map(int, input().split())
xB, yB = map(int, input().split())
xC, yC = map(int, input().split())

# Check if the triangle is a right triangle
if is_right_triangle(xA, yA, xB, yB, xC, yC):
    print("Yes")
else:
    print("No")