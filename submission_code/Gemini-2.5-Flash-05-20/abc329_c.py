# YOUR CODE HERE
import sys
from collections import defaultdict

def solve():
    # Read N, the length of the string.
    N = int(sys.stdin.readline())
    # Read S, the string itself. .strip() removes trailing newline.
    S = sys.stdin.readline().strip()

    # max_len_for_char will store the maximum length of a consecutive run
    # found for each character in S.
    # For example, if S contains "aaabaa", the longest run of 'a' is "aaa" (length 3),
    # so max_len_for_char['a'] will be 3. The longest run of 'b' is "b" (length 1),
    # so max_len_for_char['b'] will be 1.
    # The sum of these maximum lengths for all characters 'a' through 'z'
    # gives the total number of unique repetition substrings.
    # If max_len_for_char['c'] is L, it implies that 'c', 'cc', ..., 'c' repeated L times
    # are all present as unique substrings in S.
    max_len_for_char = defaultdict(int)

    # Constraints state 1 <= N, so S is guaranteed to be non-empty.
    # Initialize for the first character of the string.
    current_char = S[0]
    current_run_length = 1

    # Iterate through the string starting from the second character.
    for i in range(1, N):
        char = S[i]
        if char == current_char:
            # If the current character is the same as the one in the ongoing run,
            # extend the run length.
            current_run_length += 1
        else:
            # If the current character is different, the previous run has ended.
            # Update the maximum run length found so far for 'current_char'.
            max_len_for_char[current_char] = max(max_len_for_char[current_char], current_run_length)
            
            # Start a new run with the current character.
            current_char = char
            current_run_length = 1
    
    # After the loop finishes, there is one final run (the one ending at S[N-1])
    # whose length needs to be considered and used to update max_len_for_char.
    max_len_for_char[current_char] = max(max_len_for_char[current_char], current_run_length)

    # Calculate the total count of unique repetition substrings.
    # This is the sum of all maximum run lengths stored in max_len_for_char.
    total_unique_repetitions = 0
    for char_code in range(ord('a'), ord('z') + 1):
        c = chr(char_code)
        total_unique_repetitions += max_len_for_char[c]
    
    # Print the final result.
    print(total_unique_repetitions)

# Call the solve function to execute the program.
solve()