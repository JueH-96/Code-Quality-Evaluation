# YOUR CODE HERE

def is_palindrome(s):
    return s == s[::-1]

def solve():
    n = int(input())
    strings = [input() for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i != j and is_palindrome(strings[i] + strings[j]):
                print("Yes")
                return
    print("No")

solve()