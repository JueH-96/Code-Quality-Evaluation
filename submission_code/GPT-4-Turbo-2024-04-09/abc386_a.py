from collections import Counter

def can_form_full_house():
    import sys
    input = sys.stdin.read
    data = list(map(int, input().split()))
    
    # Count the occurrences of each number
    counts = Counter(data)
    
    # Possible counts of cards after adding one more card
    possible_counts = [v + 1 for v in counts.values()] + [1]  # +1 for each existing and 1 for a new card
    
    # Check the possible scenarios to form a full house
    for new_count in possible_counts:
        temp_counts = list(counts.values()) + [0]  # Add zero for the case of adding a new distinct card
        temp_counts[temp_counts.index(max(temp_counts))] -= 1  # Simulate adding one card to the max count
        temp_counts.append(new_count)  # Add the new count
        
        # Check if we can form a full house
        if 3 in temp_counts and 2 in temp_counts:
            print("Yes")
            return
    
    print("No")

can_form_full_house()