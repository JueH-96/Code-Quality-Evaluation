def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return

    # Parse inputs
    N, T, A = map(int, input_data)
    
    # Calculate the number of votes remaining
    remaining = N - T - A

    # Check if Takahashi's lead is secure: Even if Aoki gets all remaining votes, Takahashi would win.
    if T > A + remaining:
        print("Yes")
    # Similarly, check if Aoki's lead is secure:
    elif A > T + remaining:
        print("Yes")
    else:
        print("No")
        
if __name__ == '__main__':
    main()