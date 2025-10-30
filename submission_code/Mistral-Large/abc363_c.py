import sys
from itertools import permutations

def is_palindrome(s, K):
    return s == s[::-1] and len(s) == K

def contains_palindrome(s, K):
    for i in range(len(s) - K + 1):
        if is_palindrome(s[i:i+K], K):
            return True
    return False

def count_non_palindromic_permutations(N, K, S):
    perms = set(permutations(S))
    count = 0
    for perm in perms:
        if not contains_palindrome(''.join(perm), K):
            count += 1
    return count

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    K = int(data[1])
    S = data[2]

    result = count_non_palindromic_permutations(N, K, S)
    print(result)

if __name__ == "__main__":
    main()