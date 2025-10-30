import sys
import threading

def main():
    data = sys.stdin.buffer.read().split()
    n = int(data[0])
    A = list(map(int, data[1:]))

    total_sum = sum(A)
    k, r = divmod(total_sum, n)

    # sum of |A_i - k|
    abs_diff_sum = 0
    for x in A:
        abs_diff_sum += abs(x - k)

    # The minimum number of operations is (sum |A_i - k| - r) / 2
    print((abs_diff_sum - r) // 2)

if __name__ == "__main__":
    main()