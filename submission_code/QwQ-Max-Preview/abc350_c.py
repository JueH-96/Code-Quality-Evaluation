n = int(input())
a = list(map(int, input().split()))
A = [0] + a  # 1-based indexing
pos = [0] * (n + 2)  # pos[x] stores the current index of x

for i in range(1, n + 1):
    pos[A[i]] = i

swaps = []

for i in range(1, n + 1):
    if A[i] != i:
        j = pos[i]
        # Swap A[i] and A[j]
        A[i], A[j] = A[j], A[i]
        # Update positions
        x = A[j]  # After swap, A[j] is the previous A[i]
        pos[i] = i
        pos[x] = j
        swaps.append((i, j))

print(len(swaps))
for swap in swaps:
    print(swap[0], swap[1])