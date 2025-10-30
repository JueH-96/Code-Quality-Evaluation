# YOUR CODE HERE
s = input()
while True:
    index = s.find('ABC')
    if index == -1:
        break
    s = s[:index] + s[index+3:]
print(s)