# Read input
A, B, C, D = map(int, input().split())

# Count frequency of each number
freq = {}
for num in [A, B, C, D]:
    freq[num] = freq.get(num, 0) + 1

def can_form_full_house():
    # If we already have 4 of a kind, we can't form a full house by adding one card
    if any(count >= 4 for count in freq.values()):
        return False
    
    # If we have 3 of a kind and a single card
    if 3 in freq.values():
        # Adding another of the single card will form a full house
        return True
    
    # If we have two pairs
    if list(freq.values()).count(2) == 2:
        # Adding another of either pair will form a full house
        return True
        
    # If we have one pair and two different singles
    if list(freq.values()).count(2) == 1:
        # Adding another of either single card or the pair will form a full house
        return True
        
    return False

print("Yes" if can_form_full_house() else "No")