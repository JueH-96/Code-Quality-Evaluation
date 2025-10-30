from itertools import permutations
from collections import Counter

def is_palindrome(s, k):
    """Check if a string contains a palindrome of length k as a substring."""
    for i in range(len(s) - k + 1):
        if s[i:i+k] == s[i:i+k][::-1]:
            return True
    return False

def solve(n, k, s):
    """Count the number of strings obtained by permuting s that do not contain a palindrome of length k as a substring."""
    count = 0
    # Generate all permutations of s
    for p in set(permutations(s)):
        # Check if the permutation contains a palindrome of length k as a substring
        if not is_palindrome(''.join(p), k):
            count += 1
    return count

def main():
    n, k = map(int, input().split())
    s = input().strip()
    result = solve(n, k, s)
    print(result)

if __name__ == "__main__":
    main()