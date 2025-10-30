import sys
import bisect
from collections import defaultdict

def main():
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    pos_dict = defaultdict(list)
    for idx, val in enumerate(A):
        pos_dict[val].append(idx + 1)  # 1-based indexing
    
    # Compute sum_k
    last_occurrence = {}
    sum_k = 0
    for i in range(1, N+1):
        val = A[i-1]
        prev = last_occurrence.get(val, 0)
        contrib = (i - prev) * (N - i + 1)
        sum_k += contrib
        last_occurrence[val] = i
    
    # Compute sum_m
    sum_m = 0
    for x in range(1, N):
        if (x+1) not in pos_dict or x not in pos_dict:
            continue
        X = pos_dict[x]
        Y = pos_dict[x+1]
        prev_occurrence = 0  # previous occurrence of x+1 in the loop
        for y in Y:
            L_start = prev_occurrence + 1
            # Find in X the largest element <= y
            idx = bisect.bisect_right(X, y) - 1
            if idx >= 0:
                p_x = X[idx]
            else:
                p_x = -1
            if p_x < L_start:
                contribution = 0
            else:
                contribution = (p_x - L_start + 1) * (N - y + 1)
            sum_m += contribution
            prev_occurrence = y
    
    answer = sum_k - sum_m
    print(answer)

if __name__ == "__main__":
    main()