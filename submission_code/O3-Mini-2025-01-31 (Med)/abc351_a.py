def main():
    import sys
    data = sys.stdin.read().strip().split()
    # First 9 values are Team Takahashi's scores (A1 through A9)
    # Next 8 values are Team Aoki's scores (B1 through B8)
    A_scores = list(map(int, data[:9]))
    B_scores = list(map(int, data[9:]))
    
    takahashi_total = sum(A_scores)
    aoki_total = sum(B_scores)
    
    # To win, Team Aoki must have strictly more runs than Team Takahashi.
    # Let x be the runs in bottom of the 9th. We need:
    # aoki_total + x > takahashi_total
    # So x > takahashi_total - aoki_total, hence minimal x is (takahashi_total - aoki_total) + 1.
    required = (takahashi_total - aoki_total) + 1
    
    print(required)

if __name__ == '__main__':
    main()