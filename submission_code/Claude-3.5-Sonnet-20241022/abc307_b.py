def is_palindrome(s):
    return s == s[::-1]

N = int(input())
strings = []
for _ in range(N):
    strings.append(input())

found = False
for i in range(N):
    for j in range(N):
        if i != j:
            concat = strings[i] + strings[j]
            if is_palindrome(concat):
                found = True
                break
    if found:
        break

print("Yes" if found else "No")