# YOUR CODE HERE
import sys

def main():
    import sys
    import math
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    total_sum = sum(A)
    target_low = total_sum // N
    remainder = total_sum % N
    target_high = target_low + 1 if remainder > 0 else target_low
    count_high = remainder
    sorted_A = sorted(A)
    first_part = sorted_A[:N - count_high] if count_high else sorted_A
    last_part = sorted_A[N - count_high:] if count_high else []
    sum_increases = 0
    for x in first_part:
        if x < target_low:
            sum_increases += (target_low - x)
    for x in last_part:
        if x < target_high:
            sum_increases += (target_high - x)
    print(sum_increases)

if __name__ == "__main__":
    main()