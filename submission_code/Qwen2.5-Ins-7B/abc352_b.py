# YOUR CODE HERE
s = input()
t = input()
index = 0
result = []
for char in t:
    if index < len(s) and char == s[index]:
        result.append(index + 1)
        index += 1
    elif index > 0:
        index -= 1
print(*result)