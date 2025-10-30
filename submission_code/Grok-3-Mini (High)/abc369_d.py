import sys
data = sys.stdin.read().split()
N = int(data[0])
A = list(map(int, data[1: N + 1]))

NEG_INF = -10**18
prev_even = 0
prev_odd = NEG_INF

for S in A:
    new_even = max(prev_even, prev_odd + 2 * S)
    new_odd = max(prev_odd, prev_even + S)
    prev_even = new_even
    prev_odd = new_odd

answer = max(prev_even, prev_odd)
print(answer)