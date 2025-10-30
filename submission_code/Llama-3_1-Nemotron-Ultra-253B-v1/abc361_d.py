import sys
from collections import deque

def main():
    N = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()

    if sorted(S) != sorted(T):
        print(-1)
        return

    initial_e = N + 1
    initial_tuple = tuple(S)
    target = (N + 1, tuple(T))

    visited = set()
    queue = deque()
    queue.append((initial_e, initial_tuple, 0))
    visited.add((initial_e, initial_tuple))

    while queue:
        e, stone_tuple, steps = queue.popleft()

        if (e, stone_tuple) == target:
            print(steps)
            return

        all_colors = [None] * (N + 2)
        for i in range(1, N + 3):
            if i == e or i == e + 1:
                continue
            if i < e:
                idx = i - 1
            else:
                idx = (i - (e + 2)) + (e - 1)
            all_colors[i - 1] = stone_tuple[idx]

        for x in range(1, N + 2):
            x_plus_1 = x + 1
            if x_plus_1 > N + 2:
                continue
            if (x == e or x == e + 1) or (x_plus_1 == e or x_plus_1 == e + 1):
                continue

            color_x = all_colors[x - 1]
            color_x_plus_1 = all_colors[x]

            new_all_colors = list(all_colors)
            new_all_colors[e - 1] = color_x
            new_all_colors[e] = color_x_plus_1
            new_all_colors[x - 1] = None
            new_all_colors[x] = None

            new_tuple = []
            for i in range(1, N + 3):
                if i == x or i == x_plus_1:
                    continue
                new_tuple.append(new_all_colors[i - 1])
            new_tuple = tuple(new_tuple)
            new_e = x

            if (new_e, new_tuple) not in visited:
                visited.add((new_e, new_tuple))
                queue.append((new_e, new_tuple, steps + 1))

    print(-1)

if __name__ == "__main__":
    main()