def find_minimum_steps(S, T):
    if S == T:
        return 0, []

    from collections import deque

    queue = deque([S])
    visited = set([S])
    result = []

    while queue:
        current = queue.popleft()
        result.append(current)

        if current == T:
            break

        for i in range(len(S)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                if c != current[i]:
                    next_string = current[:i] + c + current[i + 1:]
                    if next_string not in visited:
                        visited.add(next_string)
                        queue.append(next_string)

    # Now we need to filter the result to get the path to T
    path = []
    for string in result:
        if string == T or (path and string != path[-1]):
            path.append(string)

    return len(path) - 1, path[1:]

import sys
input = sys.stdin.read
data = input().strip().split()
S = data[0]
T = data[1]

M, steps = find_minimum_steps(S, T)

print(M)
for step in steps:
    print(step)