import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    T = int(input[ptr])
    ptr += 1
    for _ in range(T):
        N = int(input[ptr])
        ptr += 1
        P = list(map(int, input[ptr:ptr + N]))
        ptr += N
        
        # Check if already sorted
        is_sorted = True
        for i in range(N):
            if P[i] != i + 1:
                is_sorted = False
                break
        if is_sorted:
            print(0)
            continue
        
        # Compute prefix sums
        prefix_sum = [0] * (N + 1)
        for i in range(1, N + 1):
            prefix_sum[i] = prefix_sum[i - 1] + P[i - 1]
        total_sum = N * (N + 1) // 2
        found = False
        
        for k in range(1, N + 1):
            # Check if P[k] == k (0-based: P[k-1] == k)
            if P[k - 1] == k:
                left_sum = prefix_sum[k - 1]
                required_left = k * (k - 1) // 2
                right_sum = prefix_sum[N] - prefix_sum[k]
                required_right = total_sum - (k * (k + 1) // 2)
                if left_sum == required_left and right_sum == required_right:
                    found = True
                    break
        print(1 if found else 2)

if __name__ == "__main__":
    main()