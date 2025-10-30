# YOUR CODE HERE
def remove_between_pipes(S):
    # Find the indices of the two '|'
    first_pipe_index = S.index('|')
    second_pipe_index = S.index('|', first_pipe_index + 1)
    
    # Remove the characters between the two '|', including the '|'
    result = S[:first_pipe_index] + S[second_pipe_index + 1:]
    
    # Print the result
    print(result)

# Read input from standard input
import sys
input = sys.stdin.read().strip()

# Call the function with the input
remove_between_pipes(input)