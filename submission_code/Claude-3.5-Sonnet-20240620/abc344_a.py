# YOUR CODE HERE
s = input().strip()
first_pipe = s.index('|')
second_pipe = s.index('|', first_pipe + 1)
result = s[:first_pipe] + s[second_pipe + 1:]
print(result)