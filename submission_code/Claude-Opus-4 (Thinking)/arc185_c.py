# YOUR CODE HERE
n, x = map(int, input().split())
a = list(map(int, input().split()))

# For each k, check if there's a pair (i, j) such that i < j < k and A[i] + A[j] = X - A[k]
for k in range(2, n):
    target = x - a[k]
    # Use hash map to find a pair that sums to target
    seen = {}
    for j in range(k):
        if target - a[j] in seen:
            i = seen[target - a[j]]
            print(i + 1, j + 1, k + 1)
            exit()
        seen[a[j]] = j

print(-1)