def main():
    T = input().strip()
    n_bags = int(input().strip())
    L = len(T)
    
    dp = [10**9] * (L + 1)
    dp[0] = 0
    
    for _ in range(n_bags):
        data = input().split()
        A_i = int(data[0])
        strings = data[1:1 + A_i]
        
        new_dp = dp[:]  # Copy current state (if we skip this bag)
        
        for j in range(L + 1):
            if dp[j] == 10**9:
                continue
            for s in strings:
                n = len(s)
                end_index = j + n
                if end_index > L:
                    continue
                if T[j:j + n] == s:
                    if dp[j] + 1 < new_dp[end_index]:
                        new_dp[end_index] = dp[j] + 1
        
        dp = new_dp  # Move to next bag with updated states
    
    print(dp[L] if dp[L] != 10**9 else -1)

if __name__ == "__main__":
    main()