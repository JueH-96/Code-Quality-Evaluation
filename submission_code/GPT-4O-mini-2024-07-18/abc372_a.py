# YOUR CODE HERE
def remove_dots(s):
    return s.replace('.', '')

if __name__ == "__main__":
    import sys
    input_string = sys.stdin.read().strip()
    result = remove_dots(input_string)
    print(result)