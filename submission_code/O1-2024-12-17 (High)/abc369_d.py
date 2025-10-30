def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # dp0 = maximum experience after processing some monsters 
    #       with an even number of defeated monsters
    # dp1 = maximum experience after processing some monsters 
    #       with an odd number of defeated monsters
    
    dp0, dp1 = 0, float('-inf')

    for x in A:
        # If we skip this monster, parity remains the same.
        # If we defeat this monster:
        #   - from dp0 -> it becomes an odd defeat -> + x
        #   - from dp1 -> it becomes an even defeat -> + 2*x
        next_dp0 = max(dp0, dp1 + 2*x)
        next_dp1 = max(dp1, dp0 + x)

        dp0, dp1 = next_dp0, next_dp1

    # We can end with either an even or odd number of total defeats.
    print(max(dp0, dp1))

# Do not forget to call main() 
if __name__ == "__main__":
    main()