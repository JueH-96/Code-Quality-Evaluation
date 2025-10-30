# YOUR CODE HERE
from collections import defaultdict

def solve():
    n, m = map(int, input().split())
    s = input()
    color_positions = defaultdict(list)
    
    for i, c in enumerate(map(int, input().split())):
        color_positions[c].append(i)
    
    result = list(s)
    
    for color in range(1, m + 1):
        positions = color_positions[color]
        length = len(positions)
        for i in range(length):
            result[positions[i]] = s[positions[(i + 1) % length]]
    
    print(''.join(result))

solve()