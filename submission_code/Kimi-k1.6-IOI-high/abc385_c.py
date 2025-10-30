import sys
from collections import defaultdict

def main():
    n = int(sys.stdin.readline())
    H = list(map(int, sys.stdin.readline().split()))
    height_pos = defaultdict(list)
    for idx in range(n):
        h = H[idx]
        height_pos[h].append(idx + 1)  # 1-based positions
    
    max_total = 1
    for positions in height_pos.values():
        k = len(positions)
        if k <= 1:
            max_total = max(max_total, k)
            continue
        dp = [{} for _ in range(k)]
        current_max = 1
        for i in range(k):
            for j in range(i):
                diff = positions[i] - positions[j]
                prev_length = dp[j].get(diff, 1)
                new_length = prev_length + 1
                if diff in dp[i]:
                    if new_length > dp[i][diff]:
                        dp[i][diff] = new_length
                else:
                    dp[i][diff] = new_length
                if new_length > current_max:
                    current_max = new_length
        max_total = max(max_total, current_max)
    print(max_total)

if __name__ == "__main__":
    main()