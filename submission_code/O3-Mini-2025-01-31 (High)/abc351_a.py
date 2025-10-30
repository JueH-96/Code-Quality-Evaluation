def main():
    import sys
    data = sys.stdin.read().split()
    # First 9 numbers are Team Takahashi's scores (top of innings)
    # Next 8 numbers are Team Aoki's scores (bottom of innings)
    a_scores = list(map(int, data[:9]))
    b_scores = list(map(int, data[9:]))
    
    total_a = sum(a_scores)
    total_b = sum(b_scores)
    
    # For Team Aoki to win, their total score must be strictly greater than Team Takahashi's.
    # Let x be the runs needed in the bottom of the 9th.
    # Then total_b + x > total_a  -->  x > total_a - total_b, so the minimum integer x is (total_a - total_b + 1).
    x = total_a - total_b + 1
    print(x)

if __name__ == '__main__':
    main()