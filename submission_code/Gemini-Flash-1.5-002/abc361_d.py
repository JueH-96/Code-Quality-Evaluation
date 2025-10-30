# YOUR CODE HERE
from collections import deque

def solve():
    n = int(input())
    s = input()
    t = input()

    q = deque([(s, 0)])
    visited = {s}

    while q:
        curr_s, dist = q.popleft()
        if curr_s == t:
            print(dist)
            return

        for i in range(len(curr_s) - 1):
            if curr_s[i] != '.' and curr_s[i+1] != '.':
                next_s = list(curr_s)
                
                first_empty = next_s.index('.')
                next_s[first_empty] = curr_s[i]
                next_s[first_empty + 1] = curr_s[i+1]
                next_s[i] = '.'
                next_s[i+1] = '.'
                next_s_str = "".join(next_s)

                if next_s_str not in visited:
                    visited.add(next_s_str)
                    q.append((next_s_str, dist + 1))

    print(-1)

solve()