# YOUR CODE HERE
S = input().strip()

stack = []
matching = {'(': ')', '[': ']', '<': '>'}

for char in S:
    if char in matching:  # Opening bracket
        stack.append(char)
    else:  # Closing bracket
        if not stack or matching[stack[-1]] != char:
            print("No")
            exit()
        stack.pop()

if len(stack) == 0:
    print("Yes")
else:
    print("No")