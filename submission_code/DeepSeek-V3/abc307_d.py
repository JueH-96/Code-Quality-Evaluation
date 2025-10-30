# YOUR CODE HERE
def process_string(N, S):
    stack = []
    to_remove = set()
    for i, char in enumerate(S):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                start = stack.pop()
                to_remove.add(start)
                to_remove.add(i)
                # Mark all characters between start and i to be removed
                for j in range(start + 1, i):
                    to_remove.add(j)
    result = []
    for i, char in enumerate(S):
        if i not in to_remove:
            result.append(char)
    return ''.join(result)

# Read input
N = int(input())
S = input().strip()

# Process the string
result = process_string(N, S)

# Print the result
print(result)