import sys
import bisect

def main():
    data = sys.stdin.read().split()
    idx = 0
    N = int(data[idx]); idx += 1
    M = int(data[idx]); idx += 1
    P = int(data[idx]); idx += 1
    A = []
    for _ in range(N):
        A.append(int(data[idx]))
        idx += 1
    B = []
    for _ in range(M):
        B.append(int(data[idx]))
        idx += 1
    A.sort()
    B.sort()
    # Compute prefix sums for B
    prefix_B = [0] * (M + 1)
    for j in range(1, M + 1):
        prefix_B[j] = prefix_B[j - 1] + B[j - 1]
    total_sum = 0
    sum_cnt = 0
    for a in A:
        x = P - a
        cnt_i = bisect.bisect_left(B, x)
        sum_B_less_x = prefix_B[cnt_i]
        total_sum += a * cnt_i + sum_B_less_x
        sum_cnt += cnt_i
    total_sum += P * (N * M - sum_cnt)
    print(total_sum)

if __name__ == '__main__':
    main()