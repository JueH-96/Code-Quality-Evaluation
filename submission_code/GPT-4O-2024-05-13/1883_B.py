# YOUR CODE HERE
def can_form_palindrome(n, k, s):
    from collections import Counter
    
    # Count the frequency of each character
    freq = Counter(s)
    
    # Count the number of characters with odd frequencies
    odd_count = sum(1 for count in freq.values() if count % 2 != 0)
    
    # To form a palindrome, we can have at most one character with an odd frequency
    # After removing k characters, the remaining length should be even if odd_count > 1
    # or odd_count should be 1 if remaining length is odd
    remaining_length = n - k
    
    if odd_count <= remaining_length and (remaining_length % 2 == odd_count % 2):
        return "YES"
    else:
        return "NO"

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
    results.append(can_form_palindrome(n, k, s))

print("
".join(results))