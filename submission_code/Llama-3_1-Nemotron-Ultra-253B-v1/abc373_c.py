import sys

n = int(sys.stdin.readline())
a = list(map(int, sys.stdin.readline().split()))
b = list(map(int, sys.stdin.readline().split()))
max_a = max(a)
max_b = max(b)
print(max_a + max_b)