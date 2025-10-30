# YOUR CODE HERE
def nand(a, b):
    return 1 if a == 0 or b == 0 else 0

N = int(input())
S = input().strip()

A = [int(c) for c in S]
result = 0
stack = []

for i in range(N):
    new_stack = [1] * (i + 1)
    for j in range(i + 1):
        if j == i:
            new_stack[j] = A[i]
        else:
            new_stack[j] = nand(stack[j], A[i])
        result += new_stack[j]
    stack = new_stack

print(result)