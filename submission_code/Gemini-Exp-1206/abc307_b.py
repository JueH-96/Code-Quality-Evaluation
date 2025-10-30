def is_palindrome(s):
    return s == s[::-1]

n = int(input())
strings = []
for _ in range(n):
    strings.append(input())

found = False
for i in range(n):
    for j in range(n):
        if i != j:
            if is_palindrome(strings[i] + strings[j]):
                found = True
                break
    if found:
        break

if found:
    print("Yes")
else:
    print("No")