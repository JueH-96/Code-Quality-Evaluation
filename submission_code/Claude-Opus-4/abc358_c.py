# YOUR CODE HERE
def solve():
    N, M = map(int, input().split())
    stands = []
    
    # Read stand information
    for i in range(N):
        s = input().strip()
        flavors = set()
        for j in range(M):
            if s[j] == 'o':
                flavors.add(j)
        stands.append(flavors)
    
    # Try all combinations of stands
    from itertools import combinations
    
    # Start with 1 stand, then 2, etc.
    for num_stands in range(1, N + 1):
        # Try all combinations of num_stands stands
        for combo in combinations(range(N), num_stands):
            # Check if this combination covers all flavors
            covered = set()
            for stand_idx in combo:
                covered.update(stands[stand_idx])
            
            # If all flavors are covered, return the number of stands
            if len(covered) == M:
                print(num_stands)
                return

solve()