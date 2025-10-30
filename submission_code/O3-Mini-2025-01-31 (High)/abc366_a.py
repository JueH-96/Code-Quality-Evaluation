def main():
    import sys
    input = sys.stdin.readline
    
    # Read the input values: total votes (N), votes for Takahashi (T), and votes for Aoki (A)
    N, T, A = map(int, input().split())
    
    # Calculate the number of remaining votes
    remaining = N - (T + A)
    
    # Determine if the election outcome is decided:
    # Takahashi's victory is guaranteed if even if Aoki gets all remaining votes, 
    # Takahashi still wins: T > A + remaining.
    # Similarly, Aoki's victory is guaranteed if even if Takahashi gets all remaining votes:
    # A > T + remaining.
    if T > A + remaining or A > T + remaining:
        print("Yes")
    else:
        print("No")

# Ensure main() is called when the program is executed.
if __name__ == '__main__':
    main()