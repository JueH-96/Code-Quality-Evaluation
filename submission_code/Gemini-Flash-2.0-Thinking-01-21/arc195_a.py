import sys

# Read N and M
N, M = map(int, sys.stdin.readline().split())

# Read sequences A and B
A = list(map(int, sys.stdin.readline().split()))
B = list(map(int, sys.stdin.readline().split()))

# Step 1: Compute first occurrences (greedy left-to-right)
# first[k] will store the smallest index i in A such that A[i] == B[k]
# and i is strictly greater than the index used for B[k-1].
first = [0] * M
current_a_idx = -1 # Index in A used for the previous element of B. Start search after this index.
for k in range(M):
    found = False
    # Search for B[k] in A starting from the index after current_a_idx
    for i in range(current_a_idx + 1, N):
        if A[i] == B[k]:
            first[k] = i
            current_a_idx = i
            found = True
            break # Found the first occurrence, move to the next element of B
    
    # If B[k] was not found after the last chosen index, no subsequence exists
    if not found:
        print("No")
        exit() # Terminate the program

# Step 2: Compute last occurrences (greedy right-to-left)
# last[k] will store the largest index i in A such that A[i] == B[k]
# and i is strictly less than the index used for B[k+1].
last = [0] * M
current_a_idx = N # Index in A used for the next element of B. Start search before this index.
for k in range(M - 1, -1, -1): # Iterate k from M-1 down to 0
    found = False
    # Search for B[k] in A starting from the index before current_a_idx
    # The range(start, stop, step) for reverse iteration:
    # start = current_a_idx - 1 (the index before the next chosen index)
    # stop = -1 (to include index 0)
    # step = -1 (for reverse iteration)
    for i in range(current_a_idx - 1, -1, -1):
        if A[i] == B[k]:
            last[k] = i
            current_a_idx = i
            found = True
            break # Found the last occurrence, move to the previous element of B
    
    # This check is mostly for robustness; if step 1 passed, a subsequence exists,
    # and finding the last occurrences from right-to-left should also succeed.
    # If it somehow fails, it implies there's no subsequence.
    if not found:
         print("No") # Output No if for some reason this step fails
         exit()

# Step 3: Compare first and last occurrences for each element of B
# If for any k, first[k] < last[k], it means B[k] could be taken from an earlier
# index (first[k]) or a later index (last[k]) while still being able to complete
# the subsequence from both ends. This implies at least two distinct subsequences exist.
# The sequence of indices (first[0], ..., first[M-1]) is one valid subsequence.
# The sequence of indices (last[0], ..., last[M-1]) is another valid subsequence.
# If first[k] < last[k] for any k, these two sequences are different.
at_least_two = False
for k in range(M):
    if first[k] < last[k]:
        at_least_two = True
        break # Found a difference, we know there are at least two

# Step 4: Output the result
if at_least_two:
    print("Yes")
else:
    print("No")