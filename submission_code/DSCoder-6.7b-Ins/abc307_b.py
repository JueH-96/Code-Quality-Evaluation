# YOUR CODE HERE

import sys

def is_palindrome(s):
    return s == s[::-1]

N = int(sys.stdin.readline().strip())
strings = [sys.stdin.readline().strip() for _ in range(N)]

for i in range(N):
    for j in range(N):
        if i != j and is_palindrome(strings[i] + strings[j]):
            print("Yes")
            sys.exit(0)

print("No")