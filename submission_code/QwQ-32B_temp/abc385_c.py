import sys
from collections import defaultdict

def main():
    N = int(sys.stdin.readline())
    H = list(map(int, sys.stdin.readline().split()))
    groups = defaultdict(list)
    for idx in range(N):
        h = H[idx]
        groups[h].append(idx)
    
    max_total = 0
    for h in groups:
        S = groups[h]
        m = len(S)
        if m == 0:
            continue
        current_max = 1
        dp = [{} for _ in range(m)]
        for j in range(m):
            current_dict = {}
            for i in range(j):
                d = S[j] - S[i]
                prev_len = dp[i].get(d, 1)
                new_len = prev_len + 1
                if d in current_dict:
                    if new_len > current_dict[d]:
                        current_dict[d] = new_len
                else:
                    current_dict[d] = new_len
            dp[j] = current_dict
            # Update current_max for this j
            if current_dict:
                cm = max(current_dict.values())
            else:
                cm = 1
            if cm > current_max:
                current_max = cm
        if current_max > max_total:
            max_total = current_max
    print(max_total)

if __name__ == "__main__":
    main()