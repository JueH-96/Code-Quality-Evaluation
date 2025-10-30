# YOUR CODE HERE
n, q = map(int, input().split())
arr = list(map(int, input().split()))

# Build initial frequency count for values 0 to n
freq = {}
for val in arr:
    if 0 <= val <= n:
        freq[val] = freq.get(val, 0) + 1

for _ in range(q):
    i, x = map(int, input().split())
    i -= 1  # Convert to 0-indexed
    
    # Update frequency count
    old_val = arr[i]
    if 0 <= old_val <= n:
        freq[old_val] -= 1
        if freq[old_val] == 0:
            del freq[old_val]
    
    arr[i] = x
    if 0 <= x <= n:
        freq[x] = freq.get(x, 0) + 1
    
    # Find MEX
    mex = 0
    while mex in freq:
        mex += 1
    
    print(mex)