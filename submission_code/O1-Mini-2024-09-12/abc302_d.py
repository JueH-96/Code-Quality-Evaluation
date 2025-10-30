import sys, bisect

def solve():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx]); idx += 1
    M = int(input[idx]); idx += 1
    D = int(input[idx]); idx += 1
    A = list(map(int, input[idx:idx+N])); idx += N
    B = list(map(int, input[idx:idx+M])); idx += M
    A.sort()
    B.sort()
    max_sum = -1
    for a in A:
        pos = bisect.bisect_right(B, a + D)
        if pos > 0 and B[pos-1] >= a - D:
            sum_val = a + B[pos-1]
            if sum_val > max_sum:
                max_sum = sum_val
    print(max_sum)