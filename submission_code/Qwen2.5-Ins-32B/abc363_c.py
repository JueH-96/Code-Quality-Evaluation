import itertools
from collections import Counter

def is_palindrome_substring(s, k):
    for i in range(len(s) - k + 1):
        if s[i:i+k] == s[i:i+k][::-1]:
            return True
    return False

def count_non_palindrome_permutations(N, K, S):
    unique_permutations = set(itertools.permutations(S))
    count = 0
    for perm in unique_permutations:
        perm_str = ''.join(perm)
        if not is_palindrome_substring(perm_str, K):
            count += 1
    return count

if __name__ == "__main__":
    N, K = map(int, input().split())
    S = input().strip()
    result = count_non_palindrome_permutations(N, K, S)
    print(result)