def is_palindrome(s):
    return s == s[::-1]

n = int(input())
strings = []
for _ in range(n):
    strings.append(input())

found = False
for i in range(n):
    if found:
        break
    for j in range(n):
        if i != j:  # distinct indices
            concatenated = strings[i] + strings[j]
            if is_palindrome(concatenated):
                found = True
                break

print("Yes" if found else "No")