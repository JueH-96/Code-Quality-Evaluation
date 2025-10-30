n = int(input())
A = list(map(int, input().split()))
A = [0] + A  # 1-based indexing

pos = [0] * (n + 1)
for i in range(1, n+1):
    pos[A[i]] = i

swap_ops = []

for i in range(1, n+1):
    if A[i] == i:
        continue
    j = pos[i]
    if i == j:
        continue  # This case shouldn't occur as A[i] != i
    # Record the swap with i < j
    if i < j:
        swap_ops.append((i, j))
    else:
        swap_ops.append((j, i))
    # Swap A[i] and A[j]
    x = A[i]
    y = A[j]
    A[i], A[j] = y, x
    # Update positions
    pos[y] = i
    pos[x] = j

# Output the result
k = len(swap_ops)
print(k)
for op in swap_ops:
    print(op[0], op[1])