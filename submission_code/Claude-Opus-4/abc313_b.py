# YOUR CODE HERE
def solve():
    N, M = map(int, input().split())
    
    # Create adjacency matrix for direct relations
    # stronger[i][j] = True means person i is stronger than person j
    stronger = [[False] * (N + 1) for _ in range(N + 1)]
    
    # Read the given relations
    for _ in range(M):
        a, b = map(int, input().split())
        stronger[a][b] = True
    
    # Compute transitive closure using Floyd-Warshall
    for k in range(1, N + 1):
        for i in range(1, N + 1):
            for j in range(1, N + 1):
                if stronger[i][k] and stronger[k][j]:
                    stronger[i][j] = True
    
    # Find potential strongest programmers
    candidates = []
    
    for person in range(1, N + 1):
        could_be_strongest = True
        
        # Check if this person could beat everyone else
        for other in range(1, N + 1):
            if person != other:
                # If we know that 'other' is stronger than 'person', 
                # then 'person' cannot be the strongest
                if stronger[other][person]:
                    could_be_strongest = False
                    break
        
        if could_be_strongest:
            # Additionally check if we have enough information
            # A person can be strongest only if for all others,
            # either we know they're weaker or it's possible they're weaker
            can_beat_all = True
            for other in range(1, N + 1):
                if person != other:
                    # If we don't know person > other and we know other > person,
                    # then person can't be strongest
                    if not stronger[person][other] and stronger[other][person]:
                        can_beat_all = False
                        break
            
            if can_beat_all:
                candidates.append(person)
    
    # Output result
    if len(candidates) == 1:
        print(candidates[0])
    else:
        print(-1)

solve()