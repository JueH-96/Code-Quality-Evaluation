# YOUR CODE HERE
s = input()
indices = []
for i in range(len(s)):
    if s[i] == '|':
        indices.append(i)
index1 = indices[0]
index2 = indices[1]
result = s[:index1] + s[index2+1:]
print(result)