from collections import deque

def min_steps(s, t):
    if s == t:
        return []

    queue = deque([(s, [s])])
    visited = set()

    while queue:
        curr, path = queue.popleft()
        if curr == t:
            return path

        visited.add(curr)

        for i in range(len(curr)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                if c != curr[i]:
                    new_s = curr[:i] + c + curr[i+1:]
                    if new_s not in visited:
                        queue.append((new_s, path + [new_s]))

    return []

def solve():
    s = input()
    t = input()

    result = min_steps(s, t)
    print(len(result))
    for step in result:
        print(step)

solve()