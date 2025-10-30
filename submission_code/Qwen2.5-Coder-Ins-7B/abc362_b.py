# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

x_A, y_A = int(data[0]), int(data[1])
x_B, y_B = int(data[2]), int(data[3])
x_C, y_C = int(data[4]), int(data[5])

def is_right_triangle(x1, y1, x2, y2, x3, y3):
    def distance(x1, y1, x2, y2):
        return (x2 - x1) ** 2 + (y2 - y1) ** 2
    
    d1 = distance(x1, y1, x2, y2)
    d2 = distance(x2, y2, x3, y3)
    d3 = distance(x3, y3, x1, y1)
    
    sides = sorted([d1, d2, d3])
    return sides[0] + sides[1] == sides[2]

if is_right_triangle(x_A, y_A, x_B, y_B, x_C, y_C):
    print("Yes")
else:
    print("No")