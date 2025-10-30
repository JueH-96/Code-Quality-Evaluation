def main():
    import sys
    from collections import defaultdict
    
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))

    MOD = 998244353
    
    # dp[i][d][l] = number of arithmetic subsequences of length l ending at index i with difference d
    # We'll store dp[i] as a dictionary { difference : [0]*(N+1) }
    # where the array index l runs from 0..N, but we only use l>=2.
    def new_list():
        return [0] * (N+1)
    
    dp = [defaultdict(new_list) for _ in range(N)]
    
    # Build all arithmetic subsequences of length >= 2
    # and keep counts by difference
    for j in range(N):
        for i in range(j):
            d = A[j] - A[i]
            dp_jd = dp[j][d]
            dp_id = dp[i][d]
            # Every pair (i, j) forms a new 2-length subsequence
            dp_jd[2] = (dp_jd[2] + 1) % MOD
            # Extend existing subsequences from dp[i][d][l] to dp[j][d][l+1]
            for l in range(2, N):
                if dp_id[l] != 0:
                    dp_jd[l+1] = (dp_jd[l+1] + dp_id[l]) % MOD
    
    # Now, count how many arithmetic subsequences of each length
    # (1-element subsequences are trivially all arithmetic, so that's N of them)
    count = [0] * (N+1)
    count[1] = N % MOD
    
    for i in range(N):
        for d, arr_l in dp[i].items():
            for l in range(2, N+1):
                if arr_l[l] != 0:
                    count[l] = (count[l] + arr_l[l]) % MOD
    
    # Print results for k = 1..N
    print(" ".join(str(count[k] % MOD) for k in range(1, N+1)))

# Call main() at the end
main()