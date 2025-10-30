def main():
    import sys
    
    A, B, C, D = map(int, sys.stdin.readline().split())
    from collections import Counter
    
    freq = Counter([A, B, C, D])
    counts = sorted(freq.values(), reverse=True)
    
    # Based on the frequency distribution of four cards,
    # determine if there's any card we can add to form a full house.
    # Full house requires a (3,2) distribution among five cards.
    
    # Possible frequency patterns among four cards are:
    # [4], [3,1], [2,2], [2,1,1], [1,1,1,1].
    # After adding one card, we can get a (3,2) arrangement only from:
    # [3,1] → add the single card again => 3,2
    # [2,2] → add one of those two integers => (3,2)
    
    if counts == [3, 1]:
        print("Yes")
    elif counts == [2, 2]:
        print("Yes")
    else:
        print("No")

main()