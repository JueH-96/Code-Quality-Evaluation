from itertools import combinations

def is_palindrome(s):
    return s == s[::-1]

def solve():
    n = int(input())
    strings = [input() for _ in range(n)]
    
    for i, j in combinations(range(n), 2):
        if is_palindrome(strings[i] + strings[j]):
            print("Yes")
            return
    
    print("No")

solve()