from collections import Counter

N, Q = map(int, input().split())
A = list(map(int, input().split()))

# Count frequencies
freq = Counter(A)

for _ in range(Q):
    i, x = map(int, input().split())
    i -= 1  # Convert to 0-indexed
    
    old_val = A[i]
    A[i] = x
    
    # Update frequency
    freq[old_val] -= 1
    if freq[old_val] == 0:
        del freq[old_val]
    freq[x] += 1
    
    # Find MEX
    mex = 0
    while mex in freq:
        mex += 1
    print(mex)