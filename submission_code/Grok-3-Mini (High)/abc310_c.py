import sys
data = sys.stdin.read().split()
N = int(data[0])
strings = data[1:N+1]
unique_count = len(set(min(s, s[::-1]) for s in strings))
print(unique_count)