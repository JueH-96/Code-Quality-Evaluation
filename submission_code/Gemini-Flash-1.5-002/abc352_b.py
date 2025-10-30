# YOUR CODE HERE
s = input()
t = input()
s_index = 0
result = []
for i, c in enumerate(t):
    if s_index < len(s) and c == s[s_index]:
        result.append(i + 1)
        s_index += 1
print(*result)