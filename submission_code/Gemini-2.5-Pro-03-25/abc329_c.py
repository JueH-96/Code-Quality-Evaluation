# YOUR CODE HERE
import sys

def solve():
    """
    Reads the input string S of length N, finds the number of unique non-empty substrings
    that are repetitions of a single character, and prints the result.
    """
    # Read input N (length of string)
    try:
        n = int(sys.stdin.readline())
    except ValueError:
        # Handle potential empty input or non-integer N if needed, although constraints suggest valid input.
        print(0)
        return
        
    # Read input string S
    s = sys.stdin.readline().strip()

    # Basic validation based on N and actual string length (optional based on problem constraints assurance)
    if n != len(s):
        # This case should not happen based on problem description but good for robustness
        # Decide how to handle: error, use len(s), etc. Let's assume input matches N.
        pass 
        
    # Base case: If the string is empty (N=0), there are no non-empty substrings.
    # Constraints state N >= 1, but handling N=0 is safe.
    if n == 0:
        print(0)
        return

    # Array to store the maximum length of consecutive runs found for each character 'a' through 'z'.
    # Initialize all maximum lengths to 0. The index corresponds to the character: 0 for 'a', 1 for 'b', etc.
    max_len = [0] * 26
    
    # We iterate through the string using a single pass to find the lengths of consecutive runs of identical characters.
    
    # Initialize tracking variables with the first character of the string.
    # `prev_char` stores the character of the current run being tracked.
    # `count` stores the length of the current run.
    prev_char = s[0]
    count = 1
    
    # Iterate from the second character (index 1) up to the end of the string (index N-1).
    for i in range(1, n):
        current_char = s[i]
        
        if current_char == prev_char:
            # If the current character is the same as the previous one,
            # it means the current run of identical characters continues. Increment the count.
            count += 1
        else:
            # If the current character is different from the previous one,
            # it signifies the end of the previous run of `prev_char`.
            # We need to update the maximum length recorded for `prev_char` if the current run's length (`count`) is greater.
            char_index = ord(prev_char) - ord('a') # Get the 0-based index for the character
            max_len[char_index] = max(max_len[char_index], count)
            
            # A new run starts with the `current_char`. Reset the tracking variables.
            prev_char = current_char
            count = 1

    # After the loop finishes, the last run of consecutive characters in the string needs to be processed.
    # This is because the update to `max_len` only happens when a character *changes*.
    # Update the maximum length for the character of this final run.
    char_index = ord(prev_char) - ord('a')
    max_len[char_index] = max(max_len[char_index], count)

    # The problem asks for the total number of unique non-empty substrings that are repetitions of a single character.
    # Consider a character 'c'. If the longest consecutive run of 'c' in S has length L,
    # then the substrings 'c', 'cc', ..., 'c' repeated L times can all be formed from this run.
    # Any run of 'c' with length l <= L will only generate substrings up to length l.
    # Therefore, the set of all unique substrings consisting only of 'c' that can be formed from S
    # is exactly {'c', 'cc', ..., 'c' repeated max_len['c'] times}.
    # The number of such unique substrings for character 'c' is precisely max_len['c'].
    # Summing these maximum lengths over all characters ('a' through 'z') gives the total count of unique single-character repetition substrings.
    total_unique_substrings = sum(max_len)
    
    # Print the final calculated total count.
    print(total_unique_substrings)

# Call the solve function to execute the logic when the script runs.
solve()

# END OF YOUR CODE HERE