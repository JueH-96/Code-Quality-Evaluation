def is_palindrome(s):
    return s == s[::-1]

def solve():
    n = int(input())
    s = [input() for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i != j:
                if is_palindrome(s[i] + s[j]):
                    print("Yes")
                    return

    print("No")

solve()