import sys
from collections import Counter

# Read all input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
Q = int(data[index])
index += 1

# Read initial array A
A = [int(data[i]) for i in range(index, index + N)]
index += N

# Create frequency counter and present set
freq = Counter(A)
present = set(k for k in freq if freq[k] > 0)

# Compute initial MEX
current_mex = 0
while current_mex in present:
    current_mex += 1

# Process each query
for _ in range(Q):
    # Read i and x for the query
    i = int(data[index])
    index += 1
    x = int(data[index])
    index += 1
    
    # Adjust index to 0-based
    idx = i - 1
    old_val = A[idx]
    new_val = x
    
    # Update the array
    A[idx] = new_val
    
    # Update frequency and present set for old value
    freq[old_val] -= 1
    removed = False
    if freq[old_val] == 0:
        del freq[old_val]
        present.remove(old_val)
        removed = True
    
    # Update frequency and present set for new value
    freq[new_val] += 1
    added = False
    if freq[new_val] == 1:
        present.add(new_val)
        added = True
    
    # Update MEX if necessary
    if removed and old_val <= current_mex:
        # MEX might decrease
        candidate = min(current_mex, old_val)
        k = candidate
        while k <= N and k in present:
            k += 1
        current_mex = k
    elif added and new_val == current_mex:
        # MEX might increase
        k = current_mex + 1
        while k <= N and k in present:
            k += 1
        current_mex = k
    
    # Print the current MEX
    print(current_mex)