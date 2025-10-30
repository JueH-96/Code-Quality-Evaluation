import collections
import sys

def solve():
    # Read the input string from standard input.
    # .strip() removes leading/trailing whitespace, including the newline character.
    S = sys.stdin.readline().strip()

    # Use collections.Counter to efficiently count character frequencies.
    # This returns a dictionary-like object where keys are characters and values are their counts.
    counts = collections.Counter(S)

    # Initialize max_freq to a value smaller than any possible frequency (e.g., -1).
    # This ensures that the first character processed will always set the initial max_freq.
    max_freq = -1
    
    # Initialize result_char. This variable will store the character that meets the criteria:
    # most frequent, and alphabetically earliest in case of a tie.
    # It will be updated as we find characters with higher frequency.
    result_char = ''

    # To handle the tie-breaking rule (alphabetical order),
    # we iterate through the unique characters of the string in alphabetical order.
    # This ensures that if multiple characters have the same maximum frequency,
    # the first one encountered (which is alphabetically smallest due to sorting)
    # will be chosen. Any subsequent characters with the same frequency but
    # alphabetically larger will not replace it.
    
    # Get all unique characters (keys) from the counts and sort them alphabetically.
    sorted_unique_chars = sorted(counts.keys())

    # Iterate through the characters in alphabetical order.
    for char in sorted_unique_chars:
        # Get the frequency of the current character.
        current_freq = counts[char]
        
        # If the current character's frequency is strictly greater than the maximum frequency found so far,
        # it means we've found a new most frequent character.
        # Update max_freq and set this character as the new result_char.
        if current_freq > max_freq:
            max_freq = current_freq
            result_char = char
        # If current_freq is equal to max_freq, we do NOT update result_char.
        # This is crucial for the alphabetical tie-breaking. Since we are iterating
        # through `sorted_unique_chars`, any `char` encountered with `current_freq == max_freq`
        # would be alphabetically *larger* than the current `result_char` (which would have
        # been set by an alphabetically smaller character with the same or higher frequency).
        # Thus, the `if` condition alone is sufficient when iterating through sorted keys.
    
    # Print the resulting character to standard output, followed by a newline.
    sys.stdout.write(result_char + '
')

# Call the solve function to execute the program.
solve()