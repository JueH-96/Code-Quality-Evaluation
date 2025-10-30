# Read input
N, X = map(int, input().split())
A = list(map(int, input().split()))

# Try all possible triples
found = False
for i in range(N-2):
    for j in range(i+1, N-1):
        for k in range(j+1, N):
            if A[i] + A[j] + A[k] == X:
                print(i+1, j+1, k+1)  # Print 1-based indices
                found = True
                break
        if found:
            break
    if found:
        break

# If no solution found
if not found:
    print(-1)