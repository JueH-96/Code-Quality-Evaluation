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
            count += (current_length * (current_length + 1)) // 2
            current_char = S[i]
            current_length = 1
    count += (current_length * (current_length + 1)) // 2
    # Now, we need to count only the unique single-character substrings
    # Since the above counts all possible substrings, but the problem requires only unique ones
    # We need to count the number of distinct characters in the string that appear at least once
    # Because for each character, there is exactly one substring that is a repetition of that character
    # So, the total unique single-character substrings is the number of distinct characters in the string
    unique_chars = set(S)
    return len(unique_chars)

# Read input
N = int(input())
S = input().strip()

# Compute the result
result = count_single_char_substrings(N, S)

# Print the result
print(result)