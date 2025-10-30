from collections import deque

def min_operations(N, S, T):
    if S == T:
        return 0

    # Check if the number of white and black stones are the same in S and T
    if S.count('W') != T.count('W') or S.count('B') != T.count('B'):
        return -1

    # Function to perform the operation
    def move(state, x):
        new_state = list(state)
        new_state[x], new_state[x+1], new_state[N], new_state[N+1] = new_state[N], new_state[N+1], new_state[x], new_state[x+1]
        return ''.join(new_state)

    # BFS to find the minimum number of operations
    queue = deque([(S, 0)])
    visited = {S}

    while queue:
        current, operations = queue.popleft()
        for x in range(N):
            if current[x] != '.' and current[x+1] != '.' and current[N] == '.' and current[N+1] == '.':
                new_state = move(current, x)
                if new_state[:N] == T:
                    return operations + 1
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_state, operations + 1))

    return -1

# Read input
N = int(input())
S = input().strip()
T = input().strip()

# Solve and print the result
print(min_operations(N, S, T))