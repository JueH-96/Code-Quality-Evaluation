# YOUR CODE HERE
import sys

def solve():
    input = sys.stdin.read
    data = input().split()
    
    n = int(data[0])
    p = list(map(int, data[1:n+1]))
    q = int(data[n+1])
    
    # Create a mapping from person number to their position in the line
    position_map = {person: idx for idx, person in enumerate(p)}
    
    queries = list(zip(map(int, data[n+2::2]), map(int, data[n+3::2])))
    
    for a, b in queries:
        pos_a = position_map[a]
        pos_b = position_map[b]
        if pos_a < pos_b:
            print(a)
        else:
            print(b)

solve()