# Read the number of elements N
N = int(input())

# Read the sequence of integers A
# The input line contains space-separated integers
A = list(map(int, input().split()))

# Flag to indicate if three consecutive equal elements are found
found_triplet = False

# Iterate through the list, checking for consecutive triplets
# We need to check for A[i], A[i+1], A[i+2]
# The index i for the first element of the triplet can range from 0 up to N-3.
# If i is N-3, the triplet is A[N-3], A[N-2], A[N-1].
# If i is N-2, we would need A[N], which is out of bounds.
# So, the loop should run for i from 0 up to N-3 inclusive.
# range(N - 2) produces indices 0, 1, ..., N-3. This is correct.
for i in range(N - 2):
    # Check if the current element and the next two are equal
    if A[i] == A[i+1] == A[i+2]:
        found_triplet = True
        # Once a triplet is found, we can stop checking
        break

# Print the final result
if found_triplet:
    print("Yes")
else:
    print("No")