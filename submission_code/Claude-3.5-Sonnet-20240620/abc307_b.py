# YOUR CODE HERE
def is_palindrome(s):
    return s == s[::-1]

n = int(input())
strings = [input().strip() for _ in range(n)]

for i in range(n):
    for j in range(i + 1, n):
        if is_palindrome(strings[i] + strings[j]) or is_palindrome(strings[j] + strings[i]):
            print("Yes")
            exit()

print("No")