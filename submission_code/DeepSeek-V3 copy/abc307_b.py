# YOUR CODE HERE
def is_palindrome(s):
    return s == s[::-1]

N = int(input())
strings = [input().strip() for _ in range(N)]

found = False
for i in range(N):
    for j in range(N):
        if i != j:
            concatenated = strings[i] + strings[j]
            if is_palindrome(concatenated):
                found = True
                break
    if found:
        break

print("Yes" if found else "No")