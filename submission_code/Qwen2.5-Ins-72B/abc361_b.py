# YOUR CODE HERE
def check_collision(a, b, c, d, e, f, g, h, i, j, k, l):
    # Check if the cuboids overlap in the x, y, and z dimensions
    x_overlap = (a < j and d > g)
    y_overlap = (b < k and e > h)
    z_overlap = (c < l and f > i)
    
    # If there is overlap in all three dimensions, the cuboids intersect with positive volume
    if x_overlap and y_overlap and z_overlap:
        print("Yes")
    else:
        print("No")

# Read input from stdin
a, b, c, d, e, f = map(int, input().split())
g, h, i, j, k, l = map(int, input().split())

# Check for collision
check_collision(a, b, c, d, e, f, g, h, i, j, k, l)