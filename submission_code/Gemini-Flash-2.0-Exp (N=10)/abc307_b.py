def is_palindrome(s):
    return s == s[::-1]

def solve():
    n = int(input())
    strings = [input() for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            if i != j:
                concatenated = strings[i] + strings[j]
                if is_palindrome(concatenated):
                    print("Yes")
                    return
    print("No")

solve()