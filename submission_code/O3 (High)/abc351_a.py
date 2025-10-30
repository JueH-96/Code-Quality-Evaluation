def main():
    import sys
    
    # Read the runs in the top of the 1st through 9th innings for Team Takahashi
    A = list(map(int, sys.stdin.readline().split()))
    # Read the runs in the bottom of the 1st through 8th innings for Team Aoki
    B = list(map(int, sys.stdin.readline().split()))
    
    # Total scores after the top of the 9th
    taka_score = sum(A)
    aoki_score = sum(B)
    
    # Team Aoki needs at least one more run than Team Takahashi
    required = (taka_score - aoki_score) + 1
    
    print(required)

# Execute main when the script is run
if __name__ == "__main__":
    main()