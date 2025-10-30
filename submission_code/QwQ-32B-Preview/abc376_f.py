def minimal_operations(N, Q, instructions):
    # Initialize positions
    L = 1
    R = 2
    operations = 0
    
    for h, T in instructions:
        if h == 'L':
            if R == T:
                # Move R to a different position
                # Choose to move R to (R + 1) % N or (R - 1) % N, select the one that's not L
                next_R = (R + 1) if (R + 1) % N != L else (R - 1) % N
                if next_R == 0:
                    next_R = N
                R = next_R
                operations += 1
            # Move L to T
            dist = min(abs(L - T), N - abs(L - T))
            operations += dist
            L = T
        else:  # h == 'R'
            if L == T:
                # Move L to a different position
                # Choose to move L to (L + 1) % N or (L - 1) % N, select the one that's not R
                next_L = (L + 1) if (L + 1) % N != R else (L - 1) % N
                if next_L == 0:
                    next_L = N
                L = next_L
                operations += 1
            # Move R to T
            dist = min(abs(R - T), N - abs(R - T))
            operations += dist
            R = T
    return operations

# Read input
import sys
input = sys.stdin.read().split()
N_Q = list(map(int, input[0].split()))
N = N_Q[0]
Q = N_Q[1]
instructions = []
for i in range(Q):
    H = input[1 + 2*i]
    T = int(input[1 + 2*i + 1])
    instructions.append((H, T))

# Compute and print the result
print(minimal_operations(N, Q, instructions))