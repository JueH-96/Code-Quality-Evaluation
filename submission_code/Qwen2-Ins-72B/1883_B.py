from collections import Counter

def can_form_palindrome(n, k, s):
    char_counts = Counter(s)
    odd_counts = sum(1 for count in char_counts.values() if count % 2 != 0)
    
    # If the number of characters with odd counts plus the number of deletions is even,
    # or if the number of deletions is greater than or equal to the number of characters with odd counts minus 1,
    # then it's possible to form a palindrome.
    return 'YES' if (odd_counts + k) % 2 == 0 or k >= odd_counts - 1 else 'NO'

# Reading the number of test cases
t = int(input())

# Processing each test case
for _ in range(t):
    n, k = map(int, input().split())
    s = input()
    print(can_form_palindrome(n, k, s))