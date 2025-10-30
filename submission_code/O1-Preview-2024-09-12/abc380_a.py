# YOUR CODE HERE
N = input().strip()
from collections import Counter
digit_counts = Counter(N)
if digit_counts.get('1', 0) == 1 and digit_counts.get('2', 0) == 2 and digit_counts.get('3', 0) == 3:
    print("Yes")
else:
    print("No")