N = int(input())
A = list(map(int, input().split()))

count = 0
# For each starting position
for l in range(N):
    # For each ending position
    for r in range(l, N):
        # Length 1 sequences are always arithmetic
        if l == r:
            count += 1
            continue
            
        # Check if sequence from l to r is arithmetic
        is_arithmetic = True
        if r-l >= 1:
            d = A[l+1] - A[l]  # Common difference
            for i in range(l+1, r):
                if A[i+1] - A[i] != d:
                    is_arithmetic = False
                    break
        if is_arithmetic:
            count += 1

print(count)