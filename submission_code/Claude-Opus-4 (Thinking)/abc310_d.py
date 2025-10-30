# YOUR CODE HERE
N, T, M = map(int, input().split())
incompatible = []
for _ in range(M):
    a, b = map(int, input().split())
    incompatible.append((a - 1, b - 1))  # Convert to 0-indexed

# Check if a subset of players can form a single team
def can_form_team(subset):
    players = []
    for i in range(N):
        if subset & (1 << i):
            players.append(i)
    
    # Check all pairs in this subset
    for i in range(len(players)):
        for j in range(i + 1, len(players)):
            for a, b in incompatible:
                if (players[i] == a and players[j] == b) or (players[i] == b and players[j] == a):
                    return False
    return True

# Dynamic programming with memoization
dp = {}

def count_ways(mask, k):
    # Base cases
    if k == 0:
        return 1 if mask == 0 else 0
    if mask == 0:
        return 0
    
    # Check memoization
    if (mask, k) in dp:
        return dp[(mask, k)]
    
    result = 0
    # Try all non-empty subsets of mask as the next team
    submask = mask
    while submask > 0:
        if can_form_team(submask):
            # Use this subset as a team, recurse on remaining players
            result += count_ways(mask ^ submask, k - 1)
        submask = (submask - 1) & mask
    
    dp[(mask, k)] = result
    return result

# Start with all players (all bits set) and T teams
print(count_ways((1 << N) - 1, T))