n = int(input())
a = list(map(int, input().split()))

stack = []

for exponent in a:
    stack.append(exponent)
    
    while len(stack) >= 2 and stack[-1] == stack[-2]:
        stack.pop()
        last_exponent = stack.pop()
        stack.append(last_exponent + 1)

print(len(stack))