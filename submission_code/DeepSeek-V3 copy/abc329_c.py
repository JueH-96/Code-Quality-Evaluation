# YOUR CODE HERE
def count_single_char_substrings(N, S):
    if N == 0:
        return 0
    count = 0
    current_char = S[0]
    current_length = 1
    for i in range(1, N):
        if S[i] == current_char:
            current_length += 1
        else:
            if current_length >= 1:
                count += 1
            current_char = S[i]
            current_length = 1
    if current_length >= 1:
        count += 1
    return count

# Read input
N = int(input())
S = input().strip()

# Compute the result
result = count_single_char_substrings(N, S)

# Print the result
print(result)