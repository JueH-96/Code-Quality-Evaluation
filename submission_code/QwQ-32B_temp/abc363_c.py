import sys
from itertools import permutations

def main():
    N, K = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()

    # Generate all unique permutations
    perms = set(permutations(S))
    unique_perms = [''.join(p) for p in perms]

    def has_palindrome(s, k):
        for i in range(len(s) - k + 1):
            substr = s[i:i+k]
            if substr == substr[::-1]:
                return True
        return False

    count = 0
    for p in unique_perms:
        if not has_palindrome(p, K):
            count += 1

    print(count)

if __name__ == "__main__":
    main()