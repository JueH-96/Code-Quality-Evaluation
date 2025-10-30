def main():
    import sys
    from collections import Counter
    
    # Read input from standard input and convert them to integers.
    cards = list(map(int, sys.stdin.read().split()))
    
    # Count the occurrences of each card value.
    freq = Counter(cards)
    
    # A full house requires two distinct integers:
    # one that appears at least three times (for the three-of-a-kind) and
    # a different one that appears at least two times (for the pair).
    for x in freq:
        for y in freq:
            if x == y:
                continue
            if freq[x] >= 3 and freq[y] >= 2:
                print("Yes")
                return
    
    print("No")
    
if __name__ == '__main__':
    main()