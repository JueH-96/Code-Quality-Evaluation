from itertools import permutations
from collections import Counter

def is_palindrome(s):
    return s == s[::-1]

def count_non_palindromic_permutations(N, K, S):
    perm_set = set(permutations(S))
    count = 0
    for perm in perm_set:
        valid = True
        perm_str = ''.join(perm)
        for i in range(N - K + 1):
            if is_palindrome(perm_str[i:i+K]):
                valid = False
                break
        if valid:
            count += 1
    return count

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    K = int(data[1])
    S = data[2]
    
    result = count_non_palindromic_permutations(N, K, S)
    print(result)

main()