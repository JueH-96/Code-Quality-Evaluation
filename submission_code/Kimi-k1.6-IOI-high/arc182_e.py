import sys
import math

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    C = int(input[ptr])
    ptr += 1
    K = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N

    def compute_sum(Y_sorted, M_val, d_val, total_terms):
        sum_total = 0
        n = len(Y_sorted)
        for i in range(n + 1):
            if i == 0:
                if not Y_sorted:
                    continue
                y0 = Y_sorted[0]
                numerator = M_val - 1 - y0
                r_min = numerator // d_val + 1
                r_min = max(r_min, 0)
                r_max = total_terms - 1
                r_min = max(r_min, 0)
                r_max = min(r_max, total_terms - 1)
                count = r_max - r_min + 1
                if count <= 0:
                    continue
                sum_contrib = (y0 - M_val) * count
                sum_r = (r_min + r_max) * count // 2
                sum_contrib += d_val * sum_r
                sum_total += sum_contrib
            elif i < n:
                prev_y = Y_sorted[i-1]
                current_y = Y_sorted[i]
                r_max_i = (M_val - 1 - prev_y) // d_val
                r_min_i = (M_val - 1 - current_y) // d_val + 1
                r_min_i = max(r_min_i, 0)
                r_max_i = min(r_max_i, total_terms - 1)
                if r_min_i > r_max_i:
                    continue
                count = r_max_i - r_min_i + 1
                if count == 0:
                    continue
                sum_contrib = (current_y - M_val) * count
                sum_r = (r_min_i + r_max_i) * count // 2
                sum_contrib += d_val * sum_r
                sum_total += sum_contrib
            else:
                if not Y_sorted:
                    continue
                last_y = Y_sorted[-1]
                r_max_i = (M_val - 1 - last_y) // d_val
                r_min_i = 0
                r_max_i = min(r_max_i, total_terms - 1)
                if r_max_i < 0:
                    continue
                count = max(0, r_max_i - r_min_i + 1)
                if count == 0:
                    continue
                sum_contrib = Y_sorted[0] * count
                sum_r = (r_min_i + r_max_i) * count // 2
                sum_contrib += d_val * sum_r
                sum_total += sum_contrib
        return sum_total

    d = math.gcd(C, M)
    M_prime = M // d
    Y_set = set(A)
    Y_sorted = sorted(Y_set)
    sum_period = compute_sum(Y_sorted, M, d, M_prime)
    full_cycles = K // M_prime
    remainder = K % M_prime
    sum_part = compute_sum(Y_sorted, M, d, remainder)
    sum_total = full_cycles * sum_period + sum_part
    print(sum_total)

if __name__ == "__main__":
    main()