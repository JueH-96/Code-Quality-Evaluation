# YOUR CODE HERE
from collections import Counter

def can_form_palindrome(s, k):
    char_count = Counter(s)
    odd_count = sum(1 for count in char_count.values() if count % 2 != 0)
    return odd_count <= k + 1 and k <= len(s) - 1

t = int(input())
for _ in range(t):
    n, k = map(int, input().split())
    s = input().strip()
    
    if can_form_palindrome(s, k):
        print("YES")
    else:
        print("NO")