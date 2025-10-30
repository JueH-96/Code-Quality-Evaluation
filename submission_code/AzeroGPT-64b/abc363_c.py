from math import factorial
from itertools import combinations

N, K = map(int, input().split())
S = input()

def all_combinations(N, K):
    """
    Generate all combinations of indices for a substring of length K in a string of length N.
    Checks for both half (if K % 2 == 0) and full length subsets.
    """
    for i in range(N - K + 1):
        for c in combinations(range(K // 2), i):
            yield c, None if K % 2 else next(iter(set(range(K // 2, K)) - set(c + c[::-1])))
            
def has_palindromesubs(s, K):
    """
    Checks if a string 's' has a palindrome substring of length 'K'.
    """
    for l, mid in all_combinations(len(s), K):
        chars = list(s)
        for idx in l:
            chars[idx], chars[K - 1 - idx] = chars[K - 1 - idx], chars[idx]
        if mid is not None:
            chars[mid] = chars[K // 2]
        return True if ''.join(chars)[:K] == ''.join(chars)[K-K:K][::-1] else False
        
def transpose(i, N):
    """
    Transforms index i, where i ranges from 0 to factorial(N) - 1,
    to lexicographical order of a permutation of the string 'S' of length N.
    """
    indices = [0] * N
    chars = sorted(S)
    for idx in range(N):
        indices[N - 1 - idx] = chars.index(S[i // factorial(N - idx - 1)])
        chars[indices[N - 1 - idx]] = None
        i %= factorial(N - idx - 1)
    return ''.join(S[i] for i in indices)
        
count = 0
for i in range(factorial(N)):
    if not has_palindromesubs(transpose(i, N), K):
        count += 1
print(count)