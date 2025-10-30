import sys
from collections import defaultdict

def main():
    MOD = 10**9 + 7
    input = sys.stdin.read().split()
    ptr = 0
    n = int(input[ptr])
    ptr += 1
    k = int(input[ptr])
    ptr += 1
    requirements = []
    for _ in range(k):
        m = int(input[ptr])
        c = int(input[ptr + 1])
        requirements.append((m, c))
        ptr += 2
    
    # Sort the checkpoints by their end index
    requirements.sort()
    # Create a dictionary to track required inversion counts for each position
    required = {}
    for m, c in requirements:
        required[m] = c
    
    # Initialize DP with base case: one permutation with 0 inversions for zero elements
    prev_dp = defaultdict(int)
    prev_dp[0] = 1
    
    # Process each element from 1 to n
    for i in range(1, n + 1):
        current_dp = defaultdict(int)
        # Each previous inversion count contributes to new counts by adding 0..i-1 new inversions
        for j_prev, count in prev_dp.items():
            for k in range(i):
                j_new = j_prev + k
                current_dp[j_new] = (current_dp[j_new] + count) % MOD
        
        # Check if current position (i-1) is a checkpoint and enforce the required inversion count
        m = i - 1
        if m in required:
            required_cnt = required[m]
            new_dp = defaultdict(int)
            new_dp[required_cnt] = current_dp.get(required_cnt, 0)
            prev_dp = new_dp
        else:
            prev_dp = current_dp
    
    # The result is the number of valid permutations that meet the requirement at the last checkpoint
    answer = prev_dp.get(required.get(n - 1, 0), 0)
    print(answer % MOD)

if __name__ == "__main__":
    main()