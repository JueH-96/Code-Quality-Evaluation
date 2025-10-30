def min_jumps(A, B):
    dx = abs(B[0] - A[0])
    dy = abs(B[1] - A[1])
    
    # If the differences have different parities, it's impossible
    if dx % 2 != dy % 2:
        return 0
    
    return max(dx, dy)

def total_distance(points):
    n = len(points)
    total = 0
    
    for i in range(n):
        for j in range(i+1, n):
            total += min_jumps(points[i], points[j])
    
    return total

# Read input
n = int(input())
points = []
for _ in range(n):
    x, y = map(int, input().split())
    points.append((x, y))

# Calculate and output the total distance
print(total_distance(points))