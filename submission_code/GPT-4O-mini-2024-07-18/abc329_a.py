# YOUR CODE HERE
def separate_characters(s):
    print(' '.join(s))

if __name__ == "__main__":
    import sys
    input_string = sys.stdin.read().strip()
    separate_characters(input_string)