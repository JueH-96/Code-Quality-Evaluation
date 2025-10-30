from collections import Counter
import math

def count_permutations(s):
    counts = Counter(s)
    n = len(s)
    denominator = 1
    for count in counts.values():
        denominator *= math.factorial(count)
    return math.factorial(n) // denominator

def contains_palindrome(s, k):
    for i in range(len(s) - k + 1):
        sub = s[i:i+k]
        is_palindrome = True
        for j in range(k // 2):
            if sub[j] != sub[k - 1 - j]:
                is_palindrome = False
                break
        if is_palindrome:
            return True
    return False

def solve():
    n, k = map(int, input().split())
    s = input()
    
    import itertools
    
    perms = set(itertools.permutations(s))
    
    count = 0
    for perm in perms:
        perm_str = "".join(perm)
        if not contains_palindrome(perm_str, k):
            count += 1
            
    print(count)

solve()