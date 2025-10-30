# Read the input values for N, L, and R
N, L, R = map(int, input().split())

# Create the initial sequence A = (1, 2, ..., N)
# list(range(1, N + 1)) generates [1, 2, ..., N]
A = list(range(1, N + 1))

# Adjust L and R for 0-based indexing used by Python lists.
# The L-th element (1-based) is at index (L-1) (0-based).
# The R-th element (1-based) is at index (R-1) (0-based).
# In Python slice A[start_index:end_index], end_index is exclusive.
# So, to include the element at 0-based index (R-1), the slice end must be (R-1)+1 = R.
start_slice_idx = L - 1
end_slice_idx = R  # This is R (1-based end position), which acts as the exclusive 0-based end for the slice

# The segment of A to be reversed is A[start_slice_idx : end_slice_idx].
# For example, if L=2, R=3: start_slice_idx=1, end_slice_idx=3. A[1:3] is elements at index 1 and 2.
# The [::-1] operation creates a reversed copy of the slice.
# This reversed copy is then assigned back to the same slice in A, modifying A in-place.
A[start_slice_idx : end_slice_idx] = A[start_slice_idx : end_slice_idx][::-1]

# Print the modified sequence A.
# The * operator unpacks the list A into individual arguments for print,
# which then prints them separated by spaces by default.
print(*A)