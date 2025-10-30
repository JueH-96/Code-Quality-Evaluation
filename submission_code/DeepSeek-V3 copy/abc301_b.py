# YOUR CODE HERE
def process_sequence(N, A):
    while True:
        # Find the first pair where the absolute difference is not 1
        found = False
        for i in range(len(A) - 1):
            if abs(A[i] - A[i+1]) != 1:
                found = True
                break
        if not found:
            break
        # Determine the range to insert
        if A[i] < A[i+1]:
            insert_list = list(range(A[i]+1, A[i+1]))
        else:
            insert_list = list(range(A[i]-1, A[i+1], -1))
        # Insert the list into the sequence
        A[i+1:i+1] = insert_list
    return A

# Read input
N = int(input())
A = list(map(int, input().split()))

# Process the sequence
result = process_sequence(N, A)

# Print the result
print(' '.join(map(str, result)))