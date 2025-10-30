import sys
data = sys.stdin.read().split()
N = int(data[0])
A = list(map(int, data[1:N+1]))
stack = []
for exp in A:
    stack.append(exp)
    while len(stack) >= 2 and stack[-1] == stack[-2]:
        exp_val = stack[-1]
        stack.pop()  # remove the last element
        stack.pop()  # remove the second last element
        stack.append(exp_val + 1)
print(len(stack))