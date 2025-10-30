# Read the input
N = int(input())
S = input()

# Initialize the count of substrings that are repetitions of one character
count = 0

# Iterate through the string
for i in range(N):
    # Check if the current character is the same as the next character
    if i == N-1 or S[i] != S[i+1]:
        # If the current character is different from the next, count the number of consecutive occurrences
        count += (i+1)*(N-i)

# Print the result
print(count)