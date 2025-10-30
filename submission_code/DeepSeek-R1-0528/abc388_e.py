n = int(input().strip())
A = list(map(int, input().split()))

j = 0
pairs = 0
for i in range(n):
    if j < n and 2 * A[j] <= A[i]:
        pairs += 1
        j += 1

print(pairs)