import sys

def main():
    data = sys.stdin.read().splitlines()
    if not data:
        return
    n = int(data[0].strip())
    graph = [[0] * n for _ in range(n)]
    index = 1
    for i in range(n-1):
        row = list(map(int, data[index].split()))
        index += 1
        for j in range(len(row)):
            graph[i][i+1+j] = row[j]
            graph[i+1+j][i] = row[j]
    
    total_states = 1 << n
    dp = [-10**18] * total_states
    dp[0] = 0
    
    for mask in range(total_states):
        if dp[mask] == -10**18:
            continue
        for i in range(n):
            if mask & (1 << i):
                continue
            for j in range(i+1, n):
                if mask & (1 << j):
                    continue
                new_mask = mask | (1 << i) | (1 << j)
                total_weight = dp[mask] + graph[i][j]
                if total_weight > dp[new_mask]:
                    dp[new_mask] = total_weight
                    
    print(max(dp))

if __name__ == "__main__":
    main()