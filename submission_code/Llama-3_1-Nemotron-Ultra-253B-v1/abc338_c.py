n = int(input())
Q = list(map(int, input().split()))
A = list(map(int, input().split()))
B = list(map(int, input().split()))

a_indices = [i for i in range(n) if A[i] > 0]
max_a = min(Q[i] // A[i] for i in a_indices) if a_indices else 0
b_indices = [i for i in range(n) if B[i] > 0]

max_s = 0
for a in range(max_a + 1):
    min_b = float('inf')
    for i in b_indices:
        available = Q[i] - a * A[i]
        min_b = min(min_b, available // B[i])
    current_s = a + min_b
    if current_s > max_s:
        max_s = current_s

print(max_s)