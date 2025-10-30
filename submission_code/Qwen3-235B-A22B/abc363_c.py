import sys
from itertools import permutations

def main():
    N, K = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()
    
    unique_perms = set()
    for p in permutations(S):
        unique_perms.add(''.join(p))
    
    count = 0
    for perm in unique_perms:
        has_palindrome = False
        for i in range(N - K + 1):
            substr = perm[i:i+K]
            if substr == substr[::-1]:
                has_palindrome = True
                break
        if not has_palindrome:
            count += 1
    print(count)

if __name__ == '__main__':
    main()