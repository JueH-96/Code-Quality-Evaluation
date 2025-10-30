# Read the number of strings
N = int(input())

# Read the strings into a list
strings = [input() for _ in range(N)]

# Find the maximum length of the strings
M = max(len(s) for s in strings)

# Initialize the vertical strings with empty strings
vertical_strings = ['' for _ in range(M)]

# Fill the vertical strings with characters from the input strings or '*'
for i in range(M):
    for j in range(N):
        if i < len(strings[N - j - 1]):
            vertical_strings[i] += strings[N - j - 1][i]
        else:
            vertical_strings[i] += '*'

# Remove trailing '*' from the vertical strings
for i in range(M):
    vertical_strings[i] = vertical_strings[i].rstrip('*')

# Print the vertical strings
for vertical_string in vertical_strings:
    print(vertical_string)