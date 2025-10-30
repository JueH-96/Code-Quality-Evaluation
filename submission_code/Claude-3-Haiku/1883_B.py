from collections import Counter

def can_form_palindrome(s, k):
    n = len(s)
    if n <= k:
        return "YES"
    
    char_count = Counter(s)
    odd_count = sum(1 for count in char_count.values() if count % 2 == 1)
    
    if odd_count > k:
        return "NO"
    
    return "YES"

t = int(input())
for _ in range(t):
    n, k = map(int, input().split())
    s = input()
    print(can_form_palindrome(s, k))