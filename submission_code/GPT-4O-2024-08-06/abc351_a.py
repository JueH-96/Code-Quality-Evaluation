# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().strip().split()
    
    # First 9 numbers are A_i
    A = list(map(int, data[:9]))
    # Next 8 numbers are B_j
    B = list(map(int, data[9:]))
    
    # Calculate total runs for Team Takahashi
    total_takahashi = sum(A)
    # Calculate total runs for Team Aoki
    total_aoki = sum(B)
    
    # Team Aoki needs to score at least one more run than Team Takahashi
    # to win the game.
    runs_needed = total_takahashi - total_aoki + 1
    
    print(runs_needed)

if __name__ == "__main__":
    main()