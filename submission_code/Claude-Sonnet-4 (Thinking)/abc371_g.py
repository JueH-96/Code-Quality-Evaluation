n = int(input())
P = list(map(int, input().split()))
A = list(map(int, input().split()))

# Convert to 0-indexed
P = [p - 1 for p in P]

original_A = A[:]
min_state = A[:]

while True:
    # Apply the operation: A_i := A_{P_i} for all i
    new_A = [A[P[i]] for i in range(n)]
    A = new_A
    
    if A == original_A:
        break
    
    if A < min_state:
        min_state = A[:]

print(' '.join(map(str, min_state)))