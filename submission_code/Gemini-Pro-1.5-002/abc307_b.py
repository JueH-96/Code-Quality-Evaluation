# YOUR CODE HERE
def is_palindrome(s):
    return s == s[::-1]

n = int(input())
strings = [input() for _ in range(n)]

found = False
for i in range(n):
    for j in range(n):
        if i != j:
            combined = strings[i] + strings[j]
            if is_palindrome(combined):
                found = True
                break
    if found:
        break

if found:
    print("Yes")
else:
    print("No")