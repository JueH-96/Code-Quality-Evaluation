import sys

def solve():
    N = int(sys.stdin.readline())
    
    all_people_info = []
    for i in range(N):
        C = int(sys.stdin.readline())
        A = list(map(int, sys.stdin.readline().split()))
        all_people_info.append((C, A))
        
    X = int(sys.stdin.readline())
    
    # List to store (bet_count, person_index) for people who bet on X
    candidates_info = []
    
    # Iterate through each person
    for i in range(N):
        bet_count, outcomes = all_people_info[i]
        
        # Check if person i+1 bet on outcome X
        if X in outcomes:
            # If they did, add them to candidates
            candidates_info.append((bet_count, i + 1)) # Store 1-based index
            
    # Find the minimum bet count among candidates
    min_bets = float('inf')
    if candidates_info: # Only calculate min_bets if there are candidates
        min_bets = min(count for count, index in candidates_info)
        
    # Identify the winners (people who bet on X with the minimum number of bets)
    winners_indices = []
    for count, index in candidates_info:
        if count == min_bets:
            winners_indices.append(index)
            
    # Sort the winner indices in ascending order
    winners_indices.sort()
    
    # Print the output
    print(len(winners_indices))
    if winners_indices:
        print(*winners_indices)

solve()