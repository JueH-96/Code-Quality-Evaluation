# YOUR CODE HERE

N = int(input())
S = input()

stack = []
for i in range(N):
    if S[i] == ')':
        tmp = ''
        while stack[-1] != '(':
            tmp += stack.pop()
        stack.pop()  # remove '('
        stack.append(tmp[::-1])
    else:
        stack.append(S[i])

print(''.join(stack))