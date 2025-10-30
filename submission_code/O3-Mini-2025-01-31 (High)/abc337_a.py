def main():
    import sys
    input = sys.stdin.readline

    # Read number of matches
    N = int(input().strip())
    
    takahashi_total = 0
    aoki_total = 0
    
    # Loop through each match and accumulate scores
    for _ in range(N):
        x, y = map(int, input().split())
        takahashi_total += x
        aoki_total += y

    # Determine and print the result based on the totals
    if takahashi_total > aoki_total:
        print("Takahashi")
    elif takahashi_total < aoki_total:
        print("Aoki")
    else:
        print("Draw")

# Call the main function so that the script runs
if __name__ == "__main__":
    main()