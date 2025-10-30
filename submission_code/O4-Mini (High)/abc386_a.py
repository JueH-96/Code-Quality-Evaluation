def main():
    import sys
    from collections import Counter
    
    # Read four cards
    A, B, C, D = map(int, sys.stdin.readline().split())
    
    # Count frequencies of each rank
    cnt = Counter([A, B, C, D])
    freqs = sorted(cnt.values())
    
    # If current frequencies are [1,3] (one card of one rank, three of another)
    # or [2,2] (two cards of one rank, two of another),
    # we can add one more to make [2,3] or [3,2] respectively.
    if freqs == [1, 3] or freqs == [2, 2]:
        print("Yes")
    else:
        print("No")

# Call main to execute
main()