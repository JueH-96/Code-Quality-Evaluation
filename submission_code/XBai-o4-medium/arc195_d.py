import sys
import bisect
from collections import defaultdict

def main():
    input = sys.stdin.read().split()
    ptr = 0
    T = int(input[ptr])
    ptr += 1
    results = []
    for _ in range(T):
        N = int(input[ptr])
        ptr += 1
        A = list(map(int, input[ptr:ptr+N]))
        ptr += N
        
        # Preprocess positions and prefix sums
        positions = defaultdict(list)
        for idx, val in enumerate(A):
            positions[val].append(idx)
        prefix_sums = dict()
        for v in positions:
            lst = positions[v]
            prefix = [0] * (len(lst) + 1)
            for i in range(len(lst)):
                prefix[i+1] = prefix[i] + lst[i]
            prefix_sums[v] = prefix
        
        # Get all unique values
        unique_values = list(positions.keys())
        
        # Initialize dp
        dp = [0] * (N + 1)
        for i in range(N-1, -1, -1):
            # Option 1: delete current element alone
            dp[i] = 1 + dp[i+1]
            # Option 2: consider other values
            for v in unique_values:
                lst = positions[v]
                if not lst:
                    continue
                j_start = bisect.bisect_left(lst, i)
                if j_start >= len(lst):
                    continue
                m = len(lst) - j_start
                sum_p = prefix_sums[v][-1] - prefix_sums[v][j_start]
                sum_ik = i * m + m * (m - 1) // 2
                swaps = sum_p - sum_ik
                new_pos = i + m
                if new_pos > N:
                    continue
                candidate = swaps + 1 + dp[new_pos]
                if candidate < dp[i]:
                    dp[i] = candidate
        results.append(dp[0])
    print('
'.join(map(str, results)))

if __name__ == "__main__":
    main()