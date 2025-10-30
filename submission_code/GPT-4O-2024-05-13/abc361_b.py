# YOUR CODE HERE
import sys
input = sys.stdin.read

def cuboids_intersect(a, b, c, d, e, f, g, h, i, j, k, l):
    # Check if there is no overlap in x, y, or z dimensions
    if d <= g or j <= a or e <= h or k <= b or f <= i or l <= c:
        return "No"
    return "Yes"

data = input().strip().split()
a, b, c, d, e, f = map(int, data[:6])
g, h, i, j, k, l = map(int, data[6:])

print(cuboids_intersect(a, b, c, d, e, f, g, h, i, j, k, l))