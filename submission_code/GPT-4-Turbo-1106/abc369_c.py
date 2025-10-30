def count_arithmetic_subsequences(N, A):
    count = 0
    i = 0
    while i < N:
        # Start of a potential arithmetic subsequence
        j = i + 1
        while j < N and A[j] - A[j - 1] == A[i + 1] - A[i]:
            j += 1
        # Length of the arithmetic subsequence
        length = j - i
        # Count the number of subsequences in this arithmetic subsequence
        count += (length * (length + 1)) // 2
        i = j - 1
    return count

# Read input from stdin
N = int(input().strip())
A = list(map(int, input().strip().split()))

# Calculate and print the answer
print(count_arithmetic_subsequences(N, A))