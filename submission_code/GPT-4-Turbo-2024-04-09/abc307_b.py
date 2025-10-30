def is_palindrome(s):
    return s == s[::-1]

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
strings = data[1:]

found = False

for i in range(N):
    for j in range(N):
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