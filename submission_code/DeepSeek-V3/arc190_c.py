import sys
from collections import defaultdict

def solve():
    N = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()
    
    # Check if possible
    # For each character in S, it must map to exactly one character in T, and vice versa (i.e., the mapping must be bijective)
    # So for every s_char, t_char must be unique, and for every t_char, s_char must be unique in the inverse mapping.
    forward = dict()
    backward = dict()
    possible = True
    for s_char, t_char in zip(S, T):
        if s_char in forward:
            if forward[s_char] != t_char:
                possible = False
                break
        else:
            forward[s_char] = t_char
        if t_char in backward:
            if backward[t_char] != s_char:
                possible = False
                break
        else:
            backward[t_char] = s_char
    if not possible:
        print(-1)
        return
    
    # Now compute the minimal operations
    # The operations form a graph where we have edges x -> y (x must be replaced by y)
    # The minimal operations is the number of edges in the graph minus the number of trees (or similar)
    # But cycles require extra steps. For each cycle of length k, we need k+1 steps.
    # So total operations is the total edges (size of forward) minus the number of cycles (each cycle of length l counts as 1)
    visited = set()
    res = 0
    # Iterate through all keys in forward
    for char in forward:
        if char not in visited:
            if forward[char] == char:
                continue  # no operation needed
            current = char
            cycle = False
            path = []
            # Traverse the chain or cycle
            while current not in visited:
                visited.add(current)
                path.append(current)
                if forward[current] in forward:
                    current = forward[current]
                    if current in visited:
                        if current in path:
                            idx = path.index(current)
                            cycle = (len(path) - idx) >= 2
                else:
                    break
            if cycle:
                res += len(path)
            else:
                res += len(path) - 1
    print(res)

solve()