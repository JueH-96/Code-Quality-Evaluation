def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # dp0: maximum xp if an even number of monsters have been defeated so far
    # dp1: maximum xp if an odd number of monsters have been defeated so far
    dp0, dp1 = 0, float('-inf')
    
    for x in A:
        # When considering monster x:
        # If we stay in even state (dp0), 
        #   either skip current (remain dp0) or come from odd state defeating monster (dp1 + 2*x)
        new_dp0 = max(dp0, dp1 + 2 * x)
        
        # If we go to odd state (dp1),
        #   either skip current (remain dp1) or come from even state defeating monster (dp0 + x)
        new_dp1 = max(dp1, dp0 + x)
        
        dp0, dp1 = new_dp0, new_dp1

    # The result is the maximum xp achievable among both parities
    print(max(dp0, dp1))

# Don't forget to call main()!
if __name__ == "__main__":
    main()