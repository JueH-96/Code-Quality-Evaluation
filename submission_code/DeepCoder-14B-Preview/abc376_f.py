def get_adjacent(pos, N):
    if pos == 1:
        return [N, 2]
    elif pos == N:
        return [N-1, 1]
    else:
        return [pos-1, pos+1]

def main():
    import sys
    from collections import deque

    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    Q = int(input[idx])
    idx += 1

    instructions = []
    for _ in range(Q):
        H = input[idx]
        T = int(input[idx + 1])
        instructions.append((H, T))
        idx += 2

    current_L = 1
    current_R = 2
    total_steps = 0

    for H, T in instructions:
        visited = set()
        queue = deque()
        initial_state = (current_L, current_R)
        queue.append((current_L, current_R, 0))
        visited.add(initial_state)

        found = False
        while queue:
            L, R, steps = queue.popleft()

            if (H == 'L' and L == T) or (H == 'R' and R == T):
                total_steps += steps
                current_L = L
                current_R = R
                found = True
                break

            # Generate all possible moves for left hand
            for new_L in get_adjacent(L, N):
                if new_L != R:
                    new_state = (new_L, R)
                    if new_state not in visited:
                        visited.add(new_state)
                        queue.append((new_L, R, steps + 1))

            # Generate all possible moves for right hand
            for new_R in get_adjacent(R, N):
                if new_R != L:
                    new_state = (L, new_R)
                    if new_state not in visited:
                        visited.add(new_state)
                        queue.append((L, new_R, steps + 1))

        if not found:
            pass  # As per problem statement, all instructions are achievable

    print(total_steps)

if __name__ == '__main__':
    main()