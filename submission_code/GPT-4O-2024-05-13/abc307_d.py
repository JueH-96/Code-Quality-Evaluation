# YOUR CODE HERE
def remove_balanced_substrings(N, S):
    stack = []
    result = []
    
    for char in S:
        if char == ')':
            # Pop characters from stack until we find a matching '('
            while stack and stack[-1] != '(':
                result.append(stack.pop())
            if stack and stack[-1] == '(':
                stack.pop()  # Remove the matching '('
        else:
            stack.append(char)
    
    # Append remaining characters in stack to result
    result.extend(stack)
    
    print(''.join(result))

# Read input
import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
S = data[1]

remove_balanced_substrings(N, S)