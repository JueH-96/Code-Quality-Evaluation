from itertools import permutations

def has_palindrome_substring(s, k):
    n = len(s)
    for i in range(n - k + 1):
        is_palindrome = True
        for j in range(k//2):
            if s[i+j] != s[i+k-1-j]:
                is_palindrome = False
                break
        if is_palindrome:
            return True
    return False

def solve():
    N, K = map(int, input().split())
    S = input().strip()
    
    # Generate all permutations
    perms = set(permutations(S))
    
    # Count permutations without palindrome substring
    count = 0
    for p in perms:
        s = ''.join(p)
        if not has_palindrome_substring(s, K):
            count += 1
            
    print(count)

solve()