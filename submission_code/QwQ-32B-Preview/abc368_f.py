MAX_N = 100001

# Precompute proper divisors for each number
divisors = [[] for _ in range(MAX_N)]
for i in range(1, MAX_N):
    for j in range(2 * i, MAX_N, i):
        divisors[j].append(i)

# Compute Grundy numbers
grundy = [0] * MAX_N
for i in range(1, MAX_N):
    if not divisors[i]:
        grundy[i] = 0
    else:
        possible = set(grundy[d] for d in divisors[i])
        # Find the smallest non-negative integer not in possible
        grundy[i] = next(j for j in range(MAX_N) if j not in possible)

# Read input
import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
A = list(map(int, data[1:N+1]))

# Compute XOR of all grundy numbers in A
xor = 0
for a in A:
    xor ^= grundy[a]

# Determine the winner
if xor != 0:
    print("Anna")
else:
    print("Bruno")