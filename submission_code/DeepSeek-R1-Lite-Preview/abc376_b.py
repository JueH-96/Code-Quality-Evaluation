from collections import deque

def minimal_steps(current, target, blocked, N):
    if current == target:
        return 0
    visited = [False] * (N + 1)
    queue = deque()
    queue.append((current, 0))
    visited[current] = True
    while queue:
        pos, steps = queue.popleft()
        # Determine neighbors
        left = pos - 1 if pos > 1 else N
        right = pos + 1 if pos < N else 1
        for neighbor in [left, right]:
            if neighbor != blocked and not visited[neighbor]:
                if neighbor == target:
                    return steps + 1
                visited[neighbor] = True
                queue.append((neighbor, steps + 1))
    return -1  # Should not happen as instructions are achievable

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    Q = int(data[1])
    instructions = data[2:]
    left = 1
    right = 2
    total_operations = 0
    for i in range(Q):
        H = instructions[2*i]
        T = int(instructions[2*i+1])
        if H == 'L':
            steps = minimal_steps(left, T, right, N)
            total_operations += steps
            left = T
        else:  # H == 'R'
            steps = minimal_steps(right, T, left, N)
            total_operations += steps
            right = T
    print(total_operations)

if __name__ == "__main__":
    main()