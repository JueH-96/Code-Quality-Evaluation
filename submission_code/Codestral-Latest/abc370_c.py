def min_lexicographical_transform(S, T):
    from collections import deque

    def bfs(S, T):
        queue = deque([(S, [])])
        visited = set()
        visited.add(S)

        while queue:
            current, path = queue.popleft()
            if current == T:
                return path + [current]

            for i in range(len(current)):
                for char in 'abcdefghijklmnopqrstuvwxyz':
                    if char != current[i]:
                        new_str = current[:i] + char + current[i+1:]
                        if new_str not in visited:
                            visited.add(new_str)
                            queue.append((new_str, path + [current]))

        return []

    result = bfs(S, T)
    print(len(result) - 1)
    for s in result:
        print(s)

# Read input
import sys
input = sys.stdin.read
data = input().split()

S = data[0]
T = data[1]

min_lexicographical_transform(S, T)