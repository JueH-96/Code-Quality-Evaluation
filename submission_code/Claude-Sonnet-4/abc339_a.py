# YOUR CODE HERE
S = input().strip()
last_dot_index = S.rfind('.')
result = S[last_dot_index + 1:]
print(result)