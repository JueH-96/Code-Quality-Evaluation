# YOUR CODE HERE
def find_different_character_index(s):
    for i in range(len(s)):
        if s[i] != s[0]:
            return i + 1  # Return 1-based index
    return -1  # This should never happen given the problem constraints

if __name__ == "__main__":
    import sys
    input_string = sys.stdin.read().strip()
    result = find_different_character_index(input_string)
    print(result)