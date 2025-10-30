def is_palindrome(s):
    n = len(s)
    for i in range(n // 2):
        if s[i] != s[n - 1 - i]:
            return False
    return True

n = int(input())
s_list = [input() for _ in range(n)]

found = False
for i in range(n):
    for j in range(n):
        if i != j:
            combined_string = s_list[i] + s_list[j]
            if is_palindrome(combined_string):
                found = True
                break
    if found:
        break

if found:
    print("Yes")
else:
    print("No")