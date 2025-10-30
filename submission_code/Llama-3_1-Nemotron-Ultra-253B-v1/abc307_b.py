def is_palindrome(s):
    return s == s[::-1]

n = int(input())
strings = [input().strip() for _ in range(n)]

for i in range(n):
    for j in range(n):
        if i != j:
            combined = strings[i] + strings[j]
            if is_palindrome(combined):
                print("Yes")
                exit()

print("No")