# YOUR CODE HERE
import sys
from collections import Counter

def can_form_palindrome(s, k):
    freq = Counter(s)
    odd_count = sum(1 for count in freq.values() if count % 2 != 0)
    # A string can form a palindrome if the number of characters with odd frequencies is at most 1
    # after removing k characters
    return odd_count - k <= 1

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
    if can_form_palindrome(s, k):
        results.append("YES")
    else:
        results.append("NO")

print("
".join(results))