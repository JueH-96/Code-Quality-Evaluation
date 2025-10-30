import sys
data = sys.stdin.read().split()
N = int(data[0])
K = int(data[1])
A = list(map(int, data[2:]))
quotients = [num // K for num in A if num % K == 0]
print(' '.join(map(str, quotients)))