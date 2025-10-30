import sys

def main():
    data = sys.stdin.read().splitlines()
    T = data[0].strip()
    N = int(data[1])
    bags = []
    index = 2
    for i in range(N):
        parts = data[index].split()
        index += 1
        A_i = int(parts[0])
        bag_contents = parts[1:1+A_i]
        bags.append(bag_contents)
    
    n_total = len(T)
    INF = 10**9
    dp = [INF] * (n_total + 1)
    dp[0] = 0
    
    for bag in bags:
        new_dp = dp[:]
        for j in range(n_total + 1):
            if dp[j] == INF:
                continue
            for s in bag:
                L = len(s)
                next_pos = j + L
                if next_pos > n_total:
                    continue
                if T[j:next_pos] == s:
                    if dp[j] + 1 < new_dp[next_pos]:
                        new_dp[next_pos] = dp[j] + 1
        dp = new_dp
    
    result = dp[n_total]
    if result == INF:
        print(-1)
    else:
        print(result)

if __name__ == "__main__":
    main()