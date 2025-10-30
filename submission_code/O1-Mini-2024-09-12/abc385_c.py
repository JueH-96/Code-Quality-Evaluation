# YOUR CODE HERE
import sys

def main():
    import sys
    import sys
    from collections import defaultdict

    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    H = list(map(int, data[1:N+1]))

    height_to_positions = defaultdict(list)
    for idx, height in enumerate(H, start=1):
        height_to_positions[height].append(idx)

    max_ans = 1

    for positions in height_to_positions.values():
        M = len(positions)
        if M ==1:
            max_ans = max(max_ans, 1)
            continue
        pos_index_map = {pos:idx for idx,pos in enumerate(positions)}
        dp = [{} for _ in range(M)]
        current_max = 1
        for i in range(M):
            for j in range(i):
                d = positions[i] - positions[j]
                prev_pos = positions[j] - d
                if prev_pos in pos_index_map:
                    k = pos_index_map[prev_pos]
                    if d in dp[k]:
                        dp[j][d] = dp[k][d] +1
                    else:
                        dp[j][d] = 2
                else:
                    dp[j][d] = 2
                current_max = max(current_max, dp[j][d])
        max_ans = max(max_ans, current_max)
    
    print(max_ans)

if __name__ == "__main__":
    main()