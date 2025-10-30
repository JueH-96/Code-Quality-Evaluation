def can_form_palindrome_after_removal(s, k):
    from collections import Counter
    
    # Count frequency of each character
    freq = Counter(s)
    
    # Number of characters that have an odd frequency
    odd_count = sum(1 for count in freq.values() if count % 2 == 1)
    
    # Remaining length after removal
    remaining_length = len(s) - k
    
    # Check if it's possible to form a palindrome
    # A palindrome can have at most one odd frequency character if its length is odd
    # If remaining_length is even, odd_count should be 0
    # If remaining_length is odd, odd_count should be at most 1
    if odd_count > remaining_length % 2:
        return "NO"
    return "YES"

import sys
input = sys.stdin.read
data = input().split()

t = int(data[0])
index = 1
results = []

for _ in range(t):
    n = int(data[index])
    k = int(data[index + 1])
    s = data[index + 2]
    index += 3
    
    result = can_form_palindrome_after_removal(s, k)
    results.append(result)

for result in results:
    print(result)