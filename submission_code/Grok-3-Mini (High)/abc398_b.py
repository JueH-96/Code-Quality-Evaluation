import sys
import collections

# Read input from stdin
data = sys.stdin.readline().strip().split()
a = list(map(int, data))

# Count frequencies using Counter
freq = collections.Counter(a)

# Check if there is a number with frequency at least 3 and at least two numbers with frequency at least 2
if max(freq.values()) >= 3 and sum(1 for k in freq if freq[k] >= 2) >= 2:
    print("Yes")
else:
    print("No")