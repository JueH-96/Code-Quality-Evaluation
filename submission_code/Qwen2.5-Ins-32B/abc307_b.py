def is_palindrome(s):
    return s == s[::-1]

def solve():
    N = int(input())
    strings = [input().strip() for _ in range(N)]
    
    for i in range(N):
        for j in range(N):
            if i != j and is_palindrome(strings[i] + strings[j]):
                print("Yes")
                return
    print("No")

solve()