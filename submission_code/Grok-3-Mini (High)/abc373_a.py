import sys
data = sys.stdin.read().splitlines()
count = sum(1 for idx, s in enumerate(data) if len(s) == idx + 1)
print(count)