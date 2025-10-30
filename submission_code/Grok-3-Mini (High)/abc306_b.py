import sys
data = sys.stdin.read().split()
result = sum(int(bit) * (2 ** i) for i, bit in enumerate(data))
print(result)