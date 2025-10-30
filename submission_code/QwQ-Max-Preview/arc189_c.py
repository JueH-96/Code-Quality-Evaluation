def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    X = int(input[idx])
    idx += 1

    A = list(map(int, input[idx:idx+N]))
    idx += N
    B = list(map(int, input[idx:idx+N]))
    idx += N
    P = list(map(int, input[idx:idx+N]))
    idx += N
    Q = list(map(int, input[idx:idx+N]))
    idx += N

    # Precompute visited_p and visited_q to check if i is in the same cycle as X
    visited_p = [False] * (N + 1)
    current = X
    while True:
        if visited_p[current]:
            break
        visited_p[current] = True
        current = P[current - 1]

    visited_q = [False] * (N + 1)
    current = X
    while True:
        if visited_q[current]:
            break
        visited_q[current] = True
        current = Q[current - 1]

    # Check initial conditions
    possible = True
    for i in range(1, N+1):
        if A[i-1]:
            if not visited_p[i]:
                possible = False
        if B[i-1]:
            if not visited_q[i]:
                possible = False
    if not possible:
        print(-1)
        return

    # Find cycle_p
    cycle_p = []
    current = X
    visited = set()
    while current not in visited:
        visited.add(current)
        cycle_p.append(current)
        current = P[current - 1]

    # Compute steps_p
    cycle_p_length = len(cycle_p)
    steps_p = {}
    for idx_node, node in enumerate(cycle_p):
        steps_p[node] = (cycle_p_length - idx_node) % cycle_p_length

    # Find cycle_q
    cycle_q = []
    current = X
    visited = set()
    while current not in visited:
        visited.add(current)
        cycle_q.append(current)
        current = Q[current - 1]

    # Compute steps_q
    cycle_q_length = len(cycle_q)
    steps_q = {}
    for idx_node, node in enumerate(cycle_q):
        steps_q[node] = (cycle_q_length - idx_node) % cycle_q_length

    # Collect red_boxes
    red_boxes = set()
    for i in range(1, N+1):
        if A[i-1] and visited_p[i]:
            if i not in steps_p:
                continue
            try:
                start_idx = cycle_p.index(i)
            except ValueError:
                continue
            step = steps_p[i]
            path = cycle_p[start_idx : start_idx + step]
            for box in path:
                if box != X:
                    red_boxes.add(box)

    # Collect blue_boxes
    blue_boxes = set()
    for i in range(1, N+1):
        if B[i-1] and visited_q[i]:
            if i not in steps_q:
                continue
            try:
                start_idx = cycle_q.index(i)
            except ValueError:
                continue
            step = steps_q[i]
            path = cycle_q[start_idx : start_idx + step]
            for box in path:
                if box != X:
                    blue_boxes.add(box)

    total = len(red_boxes.union(blue_boxes))
    print(total if total != 0 else 0)

if __name__ == "__main__":
    main()