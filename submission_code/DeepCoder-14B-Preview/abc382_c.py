n, m = map(int, input().split())
A = list(map(int, input().split()))
min_A = [0] * (n + 1)  # Using 1-based indexing

current_min = float('inf')
for i in range(1, n + 1):
    current_min = min(current_min, A[i - 1])
    min_A[i] = current_min

B = list(map(int, input().split()))

for x in B:
    low = 1
    high = n
    ans = -1
    while low <= high:
        mid = (low + high) // 2
        if min_A[mid] <= x:
            ans = mid
            high = mid - 1
        else:
            low = mid + 1
    print(ans if ans != -1 else -1)