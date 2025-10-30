# YOUR CODE HERE
S = input().strip()

stack = []
pairs = {'(': ')', '[': ']', '<': '>'}

result = "Yes"
for char in S:
    if char in pairs:  # opening bracket
        stack.append(char)
    else:  # closing bracket
        if not stack or pairs[stack[-1]] != char:
            result = "No"
            break
        stack.pop()

if result == "Yes" and stack:
    result = "No"

print(result)