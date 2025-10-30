# YOUR CODE HERE
def is_palindrome(s):
    return s == s[::-1]

n = int(input())
s = [input() for _ in range(n)]

found = False
for i in range(n):
    for j in range(n):
        if i != j:
            if is_palindrome(s[i] + s[j]):
                found = True
                break
    if found:
        break

if found:
    print("Yes")
else:
    print("No")