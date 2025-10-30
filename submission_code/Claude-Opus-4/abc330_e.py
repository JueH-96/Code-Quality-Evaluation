# YOUR CODE HERE
N, Q = map(int, input().split())
A = list(map(int, input().split()))

# Count occurrences of each number
count = {}
for num in A:
    if num in count:
        count[num] += 1
    else:
        count[num] = 1

# Process queries
for _ in range(Q):
    i, x = map(int, input().split())
    i -= 1  # Convert to 0-indexed
    
    # Update the count for the old value
    old_val = A[i]
    count[old_val] -= 1
    if count[old_val] == 0:
        del count[old_val]
    
    # Update the array and count for the new value
    A[i] = x
    if x in count:
        count[x] += 1
    else:
        count[x] = 1
    
    # Find MEX
    mex = 0
    while mex in count:
        mex += 1
    
    print(mex)