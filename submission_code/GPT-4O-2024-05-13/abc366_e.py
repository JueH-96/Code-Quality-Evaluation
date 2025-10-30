# YOUR CODE HERE
import sys
input = sys.stdin.read

def solve():
    data = input().split()
    N = int(data[0])
    D = int(data[1])
    points = [(int(data[2 + 2 * i]), int(data[3 + 2 * i])) for i in range(N)]
    
    min_x = min(p[0] for p in points)
    max_x = max(p[0] for p in points)
    min_y = min(p[1] for p in points)
    max_y = max(p[1] for p in points)
    
    count = 0
    
    for x in range(min_x - D, max_x + D + 1):
        for y in range(min_y - D, max_y + D + 1):
            total_distance = sum(abs(x - px) + abs(y - py) for px, py in points)
            if total_distance <= D:
                count += 1
    
    print(count)

solve()