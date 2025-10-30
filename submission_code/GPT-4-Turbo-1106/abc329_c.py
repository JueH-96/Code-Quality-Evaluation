def count_repeated_substrings(N, S):
    count = 0
    i = 0
    while i < N:
        char_count = 1
        while i + 1 < N and S[i] == S[i + 1]:
            i += 1
            char_count += 1
        count += (char_count * (char_count + 1)) // 2
        i += 1
    return count

# Read input from stdin
N = int(input().strip())
S = input().strip()

# Calculate and print the result to stdout
print(count_repeated_substrings(N, S))