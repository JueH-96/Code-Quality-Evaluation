# YOUR CODE HERE
import sys

def query(i, j):
    print(f"? {i} {j}")
    sys.stdout.flush()
    return int(input())

def solve(N, L, R):
    segment_sum = [0] * (N + 1)
    for i in range(N, -1, -1):
        l = L >> i
        r = R >> i
        if l == r:
            continue
        segment_sum[i] = query(i, l)
        if l + 1 < r:
            segment_sum[i] -= segment_sum[i + 1]
            segment_sum[i] %= 100
        if (L & ((1 << i) - 1)) == 0 and (R & ((1 << i) - 1)) == (1 << i) - 1:
            break
    result = 0
    for i in range(N, -1, -1):
        if (L & ((1 << i) - 1)) == 0 and (R & ((1 << i) - 1)) == (1 << i) - 1:
            result += segment_sum[i]
            result %= 100
            L >>= 1
            R >>= 1
    print(f"! {result}")

N, L, R = map(int, input().split())
solve(N, L, R)