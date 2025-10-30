# YOUR CODE HERE

import sys

def can_form_palindrome(s, k):
    char_count = [0]*26
    for char in s:
        char_count[ord(char) - ord('a')] += 1
    odd_count = sum(count % 2 for count in char_count)
    return odd_count // 2 <= k

t = int(sys.stdin.readline())
for _ in range(t):
    n, k = map(int, sys.stdin.readline().split())
    s = sys.stdin.readline().strip()
    print('YES' if can_form_palindrome(s, k) else 'NO')