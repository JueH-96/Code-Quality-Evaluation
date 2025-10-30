# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
strings = data[1:N+1]

def is_palindrome(s):
    return s == s[::-1]

for i in range(N):
    for j in range(N):
        if i != j and is_palindrome(strings[i] + strings[j]):
            print("Yes")
            sys.exit()

print("No")