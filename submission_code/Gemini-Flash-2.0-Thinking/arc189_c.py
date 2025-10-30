from collections import deque

def solve():
    n, x = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    p = list(map(int, input().split()))
    q = list(map(int, input().split()))

    x -= 1  # Adjust to 0-based indexing

    initial_state = (tuple(a), tuple(b))

    queue = deque([(initial_state, 0)])
    visited = {initial_state}

    while queue:
        current_state, steps = queue.popleft()
        current_a, current_b = current_state

        if all(current_a[i] == 0 and current_b[i] == 0 for i in range(n) if i != x):
            print(steps)
            return

        for i in range(n):
            if current_a[i] > 0 or current_b[i] > 0:
                next_a = list(current_a)
                next_b = list(current_b)

                red_balls = current_a[i]
                blue_balls = current_b[i]

                next_a[i] -= red_balls
                next_b[i] -= blue_balls

                next_a[p[i]-1] += red_balls
                next_b[q[i]-1] += blue_balls

                next_state = (tuple(next_a), tuple(next_b))

                if next_state not in visited:
                    visited.add(next_state)
                    queue.append((next_state, steps + 1))

    print(-1)

solve()