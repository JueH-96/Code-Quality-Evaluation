N = int(input().strip())
Q = list(map(int, input().strip().split()))
A = list(map(int, input().strip().split()))
B = list(map(int, input().strip().split()))

min_A = min(Q[i] // A[i] if A[i] != 0 else float('inf') for i in range(N))
min_B = min(Q[i] // B[i] if B[i] != 0 else float('inf') for i in range(N))

print(min(min_A, min_B))