def remove_between_pipes(s):
    """
    Removes all characters between two pipes in a given string.

    Args:
        s (str): The input string containing exactly two pipes.

    Returns:
        str: The resulting string after removing characters between pipes.
    """
    # Find the indices of the two pipes
    first_pipe_index = s.find('|')
    second_pipe_index = s.rfind('|')

    # Remove the characters between the two pipes, including the pipes themselves
    result = s[:first_pipe_index] + s[second_pipe_index + 1:]

    return result

# Read the input from stdin
s = input()

# Remove the characters between the pipes and print the result
print(remove_between_pipes(s))