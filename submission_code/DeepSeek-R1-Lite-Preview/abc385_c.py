from collections import defaultdict

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    H = list(map(int, data[1:N+1]))
    
    height_groups = defaultdict(list)
    for idx, height in enumerate(H, 1):
        height_groups[height].append(idx)
    
    overall_max = 1
    for height, positions in height_groups.items():
        M = len(positions)
        if M < overall_max:
            continue
        pos_set = set(positions)
        dp = [dict() for _ in range(M)]
        current_max = 1
        for i in range(M):
            for j in range(i + 1, M):
                d = positions[j] - positions[i]
                prev_pos = positions[i] - d
                if prev_pos in pos_set:
                    if d in dp[i]:
                        dp[j][d] = dp[i][d] + 1
                    else:
                        dp[j][d] = 2
                else:
                    dp[j][d] = 2
                if dp[j][d] > current_max:
                    current_max = dp[j][d]
            if current_max > overall_max:
                overall_max = current_max
    print(overall_max)

if __name__ == "__main__":
    main()