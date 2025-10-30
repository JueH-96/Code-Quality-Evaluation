import sys

def main():
    T = sys.stdin.readline().strip()
    len_T = len(T)
    N = int(sys.stdin.readline())
    bags = []
    for _ in range(N):
        parts = sys.stdin.readline().split()
        a_i = int(parts[0])
        strings = parts[1:]
        bags.append(strings)
    
    INF = float('inf')
    prev_dp = [INF] * (len_T + 1)
    prev_dp[0] = 0
    
    for i in range(N):
        current_dp = [INF] * (len_T + 1)
        for j in range(len_T + 1):
            if prev_dp[j] == INF:
                continue
            # Option 1: Do nothing
            if current_dp[j] > prev_dp[j]:
                current_dp[j] = prev_dp[j]
            # Option 2: Take a string from current bag
            for s in bags[i]:
                s_len = len(s)
                if j + s_len > len_T:
                    continue
                if T[j:j+s_len] != s:
                    continue
                new_j = j + s_len
                if current_dp[new_j] > prev_dp[j] + 1:
                    current_dp[new_j] = prev_dp[j] + 1
        prev_dp = current_dp
    
    print(prev_dp[len_T] if prev_dp[len_T] != INF else -1)

if __name__ == "__main__":
    main()