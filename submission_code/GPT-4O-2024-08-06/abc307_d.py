# YOUR CODE HERE
def remove_valid_parentheses(N, S):
    stack = []
    to_remove = [False] * N
    
    for i, char in enumerate(S):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                open_index = stack.pop()
                to_remove[open_index] = True
                to_remove[i] = True
    
    result = []
    for i in range(N):
        if not to_remove[i]:
            result.append(S[i])
    
    print(''.join(result))

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
S = data[1]
remove_valid_parentheses(N, S)