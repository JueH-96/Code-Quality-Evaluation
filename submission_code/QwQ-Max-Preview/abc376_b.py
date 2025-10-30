from collections import deque

def compute_steps(a, b, c, N):
    if a == b:
        return 0
    visited = {a: 0}
    queue = deque([a])
    while queue:
        current = queue.popleft()
        current_steps = visited[current]
        next_pos = current + 1 if current < N else 1
        prev_pos = current - 1 if current > 1 else N
        for neighbor in [prev_pos, next_pos]:
            if neighbor == b:
                return current_steps + 1
            if neighbor != c and neighbor not in visited:
                visited[neighbor] = current_steps + 1
                queue.append(neighbor)
    return -1  # As per problem statement, unreachable case

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    Q = int(input[idx])
    idx +=1
    l = 1
    r = 2
    total = 0
    for _ in range(Q):
        H = input[idx]
        idx +=1
        T = int(input[idx])
        idx +=1
        if H == 'L':
            a = l
            c = r
            l = T
        else:
            a = r
            c = l
            r = T
        steps = compute_steps(a, T, c, N)
        total += steps
    print(total)

if __name__ == "__main__":
    main()