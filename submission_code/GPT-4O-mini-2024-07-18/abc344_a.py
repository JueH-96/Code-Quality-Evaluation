# YOUR CODE HERE
def remove_between_pipes(s):
    first_pipe = s.index('|')
    second_pipe = s.index('|', first_pipe + 1)
    return s[:first_pipe] + s[second_pipe + 1:]

if __name__ == "__main__":
    import sys
    input_string = sys.stdin.read().strip()
    result = remove_between_pipes(input_string)
    print(result)