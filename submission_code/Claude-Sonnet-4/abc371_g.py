n = int(input())
P = list(map(int, input().split()))
A = list(map(int, input().split()))

# Convert to 0-indexed for easier array access
P = [p - 1 for p in P]

# Store all possible states
states = []
current = A[:]
seen = set()

while tuple(current) not in seen:
    seen.add(tuple(current))
    states.append(current[:])
    
    # Apply the operation: A_i = A_{P_i}
    new_A = [0] * n
    for i in range(n):
        new_A[i] = current[P[i]]
    current = new_A

# Find lexicographically smallest state
min_state = min(states)

print(' '.join(map(str, min_state)))