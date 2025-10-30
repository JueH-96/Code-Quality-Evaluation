def can_form_full_house(cards):
    from collections import Counter
    
    # Count the occurrences of each card
    count = Counter(cards)
    
    # Get the frequency of the counts
    freq = list(count.values())
    
    # Check the conditions for forming a Full House
    if 3 in freq and 2 in freq:
        return True
    if freq.count(3) == 1 and freq.count(1) == 2:
        return True
    if freq.count(2) == 1 and freq.count(1) == 3:
        return True
    
    return False

def main():
    import sys
    input = sys.stdin.read
    data = input().strip().split()
    cards = list(map(int, data))
    
    # Check if we can form a Full House by adding one card
    if can_form_full_house(cards):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()