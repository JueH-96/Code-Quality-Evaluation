# YOUR CODE HERE
def is_palindrome(s):
    return s == s[::-1]

n = int(input())
strings = []
for _ in range(n):
    strings.append(input().strip())

found = False
for i in range(n):
    for j in range(n):
        if i != j:  # distinct indices
            concatenated = strings[i] + strings[j]
            if is_palindrome(concatenated):
                found = True
                break
    if found:
        break

if found:
    print("Yes")
else:
    print("No")