# YOUR CODE HERE
N = int(input())
S = input()

# Find the position where "ABC" first appears as a substring in S
position = S.find("ABC")

# Print the result
print(position + 1 if position != -1 else -1)