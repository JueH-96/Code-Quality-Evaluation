# YOUR CODE HERE
N = int(input())
S = input()

# Set to store unique substrings (represented as (character, length))
unique_substrings = set()

# Process the string to find groups of consecutive identical characters
i = 0
while i < N:
    char = S[i]
    j = i
    # Find the length of consecutive identical characters
    while j < N and S[j] == char:
        j += 1
    
    # Length of this group
    group_length = j - i
    
    # Add all possible substring lengths for this character
    for length in range(1, group_length + 1):
        unique_substrings.add((char, length))
    
    i = j

# The answer is the number of unique substrings
print(len(unique_substrings))