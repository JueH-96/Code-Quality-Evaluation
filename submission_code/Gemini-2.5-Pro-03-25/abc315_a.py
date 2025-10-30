# YOUR CODE HERE
import sys

def solve():
    # Read the input string from standard input
    S = sys.stdin.readline().strip()
    
    # Define the set of vowels for efficient lookup
    vowels = {'a', 'e', 'i', 'o', 'u'}
    
    # Build the result string by iterating through the input string
    # and appending characters that are not vowels.
    result_chars = []
    for char in S:
        if char not in vowels:
            result_chars.append(char)
            
    # Join the list of non-vowel characters into a single string
    result_string = "".join(result_chars)
    
    # Print the resulting string to standard output
    print(result_string)

# Call the solve function to execute the logic
solve()
# END OF YOUR CODE