# Read N and S from stdin
N = int(input())
S = input()

# Initialize a dictionary to store the maximum run length for each character
# Keys will be characters (lowercase English letters), values will be integers (maximum consecutive length)
# We only need to store entries for characters actually present in S.
max_run_length = {}

# Iterate through the string S to find consecutive runs of identical characters.
# Use a pointer 'i' to mark the beginning of the current run.
i = 0
while i < N:
    char = S[i]  # The character of the current run
    j = i        # Use pointer 'j' to find the end of the current run
    
    # Expand 'j' as long as the character remains the same and we are within bounds.
    while j < N and S[j] == char:
        j += 1
    
    # The current run of character 'char' is from index i up to (j - 1).
    # The length of this run is (j - i).
    current_run_length = j - i
    
    # Update the maximum run length found so far for this character 'char'.
    # If 'char' is not yet a key in the dictionary, dict.get(char, 0) returns 0,
    # ensuring that the first run length becomes the initial max length.
    max_run_length[char] = max(max_run_length.get(char, 0), current_run_length)
    
    # Move the pointer 'i' to the position where the next run begins, which is 'j'.
    i = j

# The total number of unique non-empty substrings that are repetitions of one character
# is the sum of the maximum run lengths recorded for each character.
# For example, if the maximum run length for 'a' is 3 (e.g., from "aaabaa"),
# then the unique 'a'-only substrings are 'a' (length 1), 'aa' (length 2), and 'aaa' (length 3).
# There are 3 such unique substrings.
# We sum these counts for all characters that appeared in the string.
total_count = sum(max_run_length.values())

# Print the final result to standard output.
print(total_count)