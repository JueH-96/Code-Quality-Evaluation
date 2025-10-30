# YOUR CODE HERE
def can_form_full_house(cards):
    from collections import Counter
    
    # Count the occurrences of each card
    card_count = Counter(cards)
    
    # Get the frequencies of the counts
    freq = list(card_count.values())
    
    # Check if we can form a full house by adding one card
    if 3 in freq and 1 in freq:
        # We have three of one card and one of another, adding one more of the second card forms a full house
        return "Yes"
    if 2 in freq and len(freq) == 2:
        # We have two pairs, adding one more of any of the pairs forms a full house
        return "Yes"
    
    return "No"

def main():
    import sys
    input = sys.stdin.read
    data = input().strip()
    
    # Parse the input
    A, B, C, D = map(int, data.split())
    
    # Check if a full house can be formed
    result = can_form_full_house([A, B, C, D])
    
    # Print the result
    print(result)

if __name__ == "__main__":
    main()