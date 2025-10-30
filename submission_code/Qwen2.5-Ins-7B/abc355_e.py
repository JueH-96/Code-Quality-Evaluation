# YOUR CODE HERE
import sys

def query(i, j):
    print(f'? {i} {j}', flush=True)
    return int(sys.stdin.readline().strip())

def solve(N, L, R):
    total_sum = 0
    i = 0
    while 2**i <= N:
        j = (L // (2**i)) - (L % (2**i) == 0)
        if j < 0:
            j = 0
        r = 2**i * (j + 1) - 1
        if r > R:
            r = R
        if r >= L:
            total_sum += query(i, j) * (min(r, R) - max(L, 2**i * j) + 1)
        i += 1
    print(f'! {total_sum % 100}', flush=True)

N, L, R = map(int, input().split())
solve(N, L, R)