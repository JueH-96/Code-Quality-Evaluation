# YOUR CODE HERE
def remove_abc_substrings(S):
    stack = []
    
    for char in S:
        stack.append(char)
        # Check if the last three characters in the stack form "ABC"
        if len(stack) >= 3 and stack[-3] == 'A' and stack[-2] == 'B' and stack[-1] == 'C':
            # Remove the last three characters
            stack.pop()
            stack.pop()
            stack.pop()
    
    # Convert the stack back to a string
    return ''.join(stack)

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    S = input().strip()
    result = remove_abc_substrings(S)
    print(result)