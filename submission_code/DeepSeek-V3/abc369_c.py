# YOUR CODE HERE
def count_arithmetic_subsequences(N, A):
    if N == 1:
        return 1
    total = 0
    current_length = 1
    for i in range(1, N):
        if i == 1:
            d = A[i] - A[i-1]
        else:
            new_d = A[i] - A[i-1]
            if new_d == d:
                current_length += 1
            else:
                total += current_length * (current_length + 1) // 2
                current_length = 2
                d = new_d
    total += current_length * (current_length + 1) // 2
    return total

# Read input
N = int(input())
A = list(map(int, input().split()))

# Compute the result
result = count_arithmetic_subsequences(N, A)

# Print the result
print(result)