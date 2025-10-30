import sys
from collections import defaultdict

def main():
    N = int(sys.stdin.readline())
    H = list(map(int, sys.stdin.readline().split()))
    groups = defaultdict(list)
    for idx, h in enumerate(H):
        groups[h].append(idx + 1)  # 1-based positions

    max_len = 1
    for h in groups:
        pos_list = groups[h]
        m = len(pos_list)
        if m <= 1:
            continue
        dp = [{} for _ in range(m)]
        current_max = 1
        for i in range(m):
            for j in range(i):
                diff = pos_list[i] - pos_list[j]
                prev = dp[j].get(diff, 1)
                new_len = prev + 1
                dp[i][diff] = new_len
                if new_len > current_max:
                    current_max = new_len
        if current_max > max_len:
            max_len = current_max
    print(max_len)

if __name__ == "__main__":
    main()