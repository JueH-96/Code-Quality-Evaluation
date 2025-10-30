def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Parse input
    it = iter(input_data)
    N = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    # There are N-1 pairs (S_i, T_i)
    ST = [ (int(next(it)), int(next(it))) for _ in range(N-1) ]
    
    # We'll use a greedy dynamic programming approach.
    # Think of it like this:
    #   Let dp[i] be the maximum amount of money you can have in the currency of country i 
    #   after optimally combining the initial A_i and performing conversions from previous country.
    # 
    # Conversion rule: from country i to country i+1, you can convert in integer multiples.
    #   For each conversion operation, you must spend S_i units of currency i and you receive T_i units
    #   of currency i+1.
    # Since operations can be done arbitrarily in any order, you are allowed to wait 
    # until you have accumulated all the money in currency i (both your initial amount A_i and
    # the money coming from conversions from previous countries) so that you can minimize wastage
    # from odd remainders when doing integer division.
    #
    # For country 1, you start with dp[1] = A_1.
    # Then, for i from 1 to N-1:
    #    dp[i+1] = A[i+1] + (dp[i] // S_i) * T_i
    # The leftover dp[i] % S_i cannot be used (since you cannot convert a fraction of S_i),
    # so the best you can do is convert as many full packages as possible.
    
    dp = A[0]
    for i in range(N-1):
        S_i, T_i = ST[i]
        # Maximum number of times you can perform the operation with country i's currency
        packages = dp // S_i
        dp = A[i+1] + packages * T_i
    sys.stdout.write(str(dp))
    
if __name__ == '__main__':
    main()