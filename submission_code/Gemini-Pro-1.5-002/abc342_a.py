# YOUR CODE HERE
s = input()
n = len(s)
if s[0] == s[1]:
    for i in range(2, n):
        if s[i] != s[0]:
            print(i + 1)
            break
elif s[0] == s[2]:
    print(2)
else:
    print(1)