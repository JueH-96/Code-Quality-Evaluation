import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    T = int(input[ptr])
    ptr += 1
    for _ in range(T):
        N = int(input[ptr])
        ptr += 1
        P = list(map(int, input[ptr:ptr+N]))
        ptr += N
        
        # Check if already sorted
        sorted_flag = True
        for i in range(N):
            if P[i] != i + 1:
                sorted_flag = False
                break
        if sorted_flag:
            print(0)
            continue
        
        # Precompute prefix and suffix arrays
        prefix_sum = [0] * (N + 1)
        prefix_max = [0] * (N + 1)
        prefix_min = [float('inf')] * (N + 1)
        for i in range(1, N + 1):
            prefix_sum[i] = prefix_sum[i-1] + P[i-1]
            prefix_max[i] = max(prefix_max[i-1], P[i-1])
            prefix_min[i] = min(prefix_min[i-1], P[i-1])
        
        suffix_sum = [0] * (N + 1)
        suffix_max = [0] * (N + 1)
        suffix_min = [float('inf')] * (N + 1)
        for i in range(N-1, -1, -1):
            suffix_sum[i] = suffix_sum[i+1] + P[i]
            suffix_max[i] = max(suffix_max[i+1], P[i])
            suffix_min[i] = min(suffix_min[i+1], P[i])
        
        found = False
        for k in range(1, N+1):
            if P[k-1] != k:
                continue
            
            left_valid = True
            if k - 1 >= 1:
                sum_l = prefix_sum[k-1]
                max_l = prefix_max[k-1]
                min_l = prefix_min[k-1]
                expected_sum_l = (k-1) * k // 2
                if sum_l != expected_sum_l or max_l != k-1 or min_l != 1:
                    left_valid = False
            
            right_valid = True
            if k + 1 <= N:
                sum_r = suffix_sum[k]
                expected_sum_r = (N - k) * (N + k + 1) // 2
                max_r = suffix_max[k]
                min_r = suffix_min[k]
                if sum_r != expected_sum_r or max_r != N or min_r != k+1:
                    right_valid = False
            
            if left_valid and right_valid:
                found = True
                break
        
        print(1 if found else 2)

if __name__ == '__main__':
    main()