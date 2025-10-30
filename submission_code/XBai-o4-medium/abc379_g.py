import sys
from collections import defaultdict

mod = 998244353

def generate_valid_sequences(s):
    allowed = []
    for c in s:
        if c == '?':
            allowed.append([1, 2, 3])
        else:
            allowed.append([int(c)])
    current = []
    for d in allowed[0]:
        current.append((d,))
    for i in range(1, len(s)):
        next_current = []
        for seq in current:
            last = seq[-1]
            for d in allowed[i]:
                if d != last:
                    next_current.append(seq + (d,))
        current = next_current
        if not current:
            break
    return current

def main():
    H, W = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(H)]
    # Decide to transpose
    if H < W:
        # transpose
        processed_grid = [''.join(col) for col in zip(*grid)]
    else:
        processed_grid = grid
    # Generate configurations for each row
    configurations = []
    for row in processed_grid:
        configs = generate_valid_sequences(row)
        configurations.append(configs)
    # Check if any row has no configurations
    for configs in configurations:
        if not configs:
            print(0)
            return
    # DP
    current_dp = defaultdict(int)
    # Initialize with first row
    for config in configurations[0]:
        current_dp[config] = 1
    # Process remaining rows
    for i in range(1, len(processed_grid)):
        next_dp = defaultdict(int)
        curr_configs = configurations[i]
        for prev_config in current_dp:
            prev_count = current_dp[prev_config]
            for curr_config in curr_configs:
                # check compatibility
                compatible = True
                for a, b in zip(prev_config, curr_config):
                    if a == b:
                        compatible = False
                        break
                if compatible:
                    next_dp[curr_config] = (next_dp[curr_config] + prev_count) % mod
        current_dp = next_dp
        if not current_dp:
            print(0)
            return
    print(sum(current_dp.values()) % mod)

if __name__ == "__main__":
    main()