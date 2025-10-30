# Read N, K, X from the first line
N, K, X = map(int, input().split())

# Read the sequence A from the second line
A = list(map(int, input().split()))

# K is 1-based as per the problem statement.
# list.insert(index, element) inserts 'element' at 'index' (0-based).
# To insert X immediately after the K-th element (which is A[K-1] in 0-based indexing),
# X should be placed at index K.
# For example, if K=1 (insert after 1st element A[0]), X goes to index 1.
# If K=N (insert after N-th element A[N-1]), X goes to index N (appended).
A.insert(K, X)

# Print the modified sequence B (A has been modified in-place).
# The * operator unpacks the list elements as arguments to print,
# which separates them by spaces by default.
print(*A)