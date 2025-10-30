def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    T = input[ptr]
    ptr += 1
    N = int(input[ptr])
    ptr += 1
    bags = []
    for _ in range(N):
        A_i = int(input[ptr])
        ptr += 1
        strings = input[ptr:ptr+A_i]
        ptr += A_i
        bags.append(strings)
    
    len_T = len(T)
    INF = float('inf')
    prev_dp = [INF] * (len_T + 1)
    prev_dp[0] = 0
    
    for bag in bags:
        curr_dp = [INF] * (len_T + 1)
        for j in range(len_T + 1):
            if prev_dp[j] == INF:
                continue
            # Option 1: do nothing
            if curr_dp[j] > prev_dp[j]:
                curr_dp[j] = prev_dp[j]
            # Option 2: use a string from the current bag
            for s in bag:
                l = len(s)
                new_j = j + l
                if new_j > len_T:
                    continue
                if T[j:new_j] == s:
                    if curr_dp[new_j] > prev_dp[j] + 1:
                        curr_dp[new_j] = prev_dp[j] + 1
        prev_dp = curr_dp
    
    result = prev_dp[len_T] if prev_dp[len_T] != INF else -1
    print(result)

if __name__ == "__main__":
    main()