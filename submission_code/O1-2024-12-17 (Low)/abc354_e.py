def main():
    import sys
    sys.setrecursionlimit(10**7)
    
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    AB = [(int(input_data[2*i+1]), int(input_data[2*i+2])) for i in range(N)]
    
    # Precompute pairs (edges) that can be removed if both cards are in the current state
    edges = []
    for i in range(N):
        for j in range(i+1, N):
            if AB[i][0] == AB[j][0] or AB[i][1] == AB[j][1]:
                edges.append((i,j))
    
    # dp[mask] = -1 (uninitialized), 0 (losing), 1 (winning)
    dp = [-1]* (1<<N)
    
    def is_winning(mask):
        # If already computed, return it
        if dp[mask] != -1:
            return dp[mask]
        
        # Check all possible pairs that can be taken from the current mask
        for i, j in edges:
            if (mask & (1 << i)) and (mask & (1 << j)):
                # Remove the pair (i,j)
                nxt = mask ^ (1 << i) ^ (1 << j)
                
                # If the opponent is in a losing position after we remove (i,j), 
                # then current position is winning.
                if is_winning(nxt) == 0:
                    dp[mask] = 1
                    return 1
        
        # If no move leads to the opponent being in losing position, then it's losing.
        dp[mask] = 0
        return 0
    
    full_mask = (1 << N) - 1
    winner = is_winning(full_mask)
    print("Takahashi" if winner == 1 else "Aoki")

if __name__ == "__main__":
    main()