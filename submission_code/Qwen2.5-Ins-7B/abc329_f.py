# YOUR CODE HERE
from collections import defaultdict

def solve():
    n, q = map(int, input().split())
    color_count = defaultdict(int)
    box_colors = [0] * (n + 1)
    
    for i in range(1, n + 1):
        color = int(input())
        box_colors[i] = color
        color_count[color] += 1
    
    for _ in range(q):
        a, b = map(int, input().split())
        if box_colors[a] != 0:
            color_count[box_colors[a]] -= 1
            box_colors[b] = box_colors[a]
            color_count[box_colors[b]] += 1
        print(len(color_count))
        
solve()