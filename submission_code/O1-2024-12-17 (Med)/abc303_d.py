def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    X = int(input_data[0])
    Y = int(input_data[1])
    Z = int(input_data[2])
    S = input_data[3]
    n = len(S)
    
    INF = 10**20  # A sufficiently large number
    # dp0[i] = min cost to produce S[:i] with caps lock OFF
    # dp1[i] = min cost to produce S[:i] with caps lock ON
    dp0 = [INF]*(n+1)
    dp1 = [INF]*(n+1)
    
    # Initially, caps is off at cost 0, or we can toggle it on at cost Z
    dp0[0] = 0
    dp1[0] = Z
    
    for i in range(n):
        c = S[i]
        cost0 = dp0[i]
        cost1 = dp1[i]
        
        if c == 'a':
            # From caps off, press 'a'
            dp0[i+1] = min(dp0[i+1], cost0 + X)
            # From caps off, toggle on + SHIFT
            dp1[i+1] = min(dp1[i+1], cost0 + Z + Y)
            
            # From caps on, SHIFT
            dp1[i+1] = min(dp1[i+1], cost1 + Y)
            # From caps on, toggle off + press 'a'
            dp0[i+1] = min(dp0[i+1], cost1 + Z + X)
        else:  # c == 'A'
            # From caps off, SHIFT
            dp0[i+1] = min(dp0[i+1], cost0 + Y)
            # From caps off, toggle on + press 'A'
            dp1[i+1] = min(dp1[i+1], cost0 + Z + X)
            
            # From caps on, press 'A'
            dp1[i+1] = min(dp1[i+1], cost1 + X)
            # From caps on, toggle off + SHIFT
            dp0[i+1] = min(dp0[i+1], cost1 + Z + Y)
    
    # The answer is the minimum cost to have formed all characters,
    # regardless of the final caps lock state.
    print(min(dp0[n], dp1[n]))

# Do not forget to call main()!
if __name__ == "__main__":
    main()