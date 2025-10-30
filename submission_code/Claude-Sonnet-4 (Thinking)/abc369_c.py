n = int(input())
A = list(map(int, input().split()))

count = 0

for l in range(n):
    # Single element is always an arithmetic progression
    count += 1
    
    if l == n - 1:
        continue
    
    # Check for longer arithmetic progressions
    diff = A[l+1] - A[l]
    r = l + 1
    
    while r < n and A[r] - A[r-1] == diff:
        count += 1
        r += 1

print(count)