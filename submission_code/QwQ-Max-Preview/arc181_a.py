import sys
from sys import stdin

def main():
    input = sys.stdin.read().split()
    idx = 0
    T = int(input[idx])
    idx += 1
    for _ in range(T):
        N = int(input[idx])
        idx += 1
        P = list(map(int, input[idx:idx+N]))
        idx += N
        
        # Precompute prefix_max, suffix_min, suffix_max
        prefix_max = [0] * (N + 1)
        current_max = 0
        for i in range(1, N+1):
            current_max = max(current_max, P[i-1])
            prefix_max[i] = current_max
        
        suffix_min = [0] * (N + 2)
        suffix_max = [0] * (N + 2)
        current_min = float('inf')
        current_max = -float('inf')
        for i in range(N, 0, -1):
            current_min = min(current_min, P[i-1])
            current_max = max(current_max, P[i-1])
            suffix_min[i] = current_min
            suffix_max[i] = current_max
        
        # Find the largest valid k
        valid_k = -1
        for k in range(1, N+1):
            left_valid = True
            if k > 1:
                if prefix_max[k-1] != k-1:
                    left_valid = False
            right_valid = True
            if k < N:
                if suffix_min[k+1] != k+1 or suffix_max[k+1] != N:
                    right_valid = False
            if left_valid and right_valid and P[k-1] == k:
                valid_k = k
                break
        
        if valid_k == -1:
            print(1)
        else:
            # Split into left and right
            left = P[:valid_k-1]
            right = P[valid_k:]
            # Check if left and right are already correct
            left_correct = (len(left) == 0) or (sorted(left) == list(range(1, valid_k)))
            right_correct = (len(right) == 0) or (sorted(right) == list(range(valid_k+1, N+1)))
            if left_correct and right_correct:
                print(1)
            else:
                print(1 + (0 if left_correct else 1) + (0 if right_correct else 1))

if __name__ == '__main__':
    main()