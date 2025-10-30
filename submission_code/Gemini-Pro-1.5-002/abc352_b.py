# YOUR CODE HERE
s = input()
t = input()
s_ptr = 0
result = []
for i in range(len(t)):
    if s_ptr < len(s) and s[s_ptr] == t[i]:
        result.append(i + 1)
        s_ptr += 1
print(*result)