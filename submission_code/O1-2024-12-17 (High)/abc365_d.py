def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    S = data[1]

    # Dictionary to get the winning move against opponent's move
    # R -> P, P -> S, S -> R
    beat = {'R': 'P', 'P': 'S', 'S': 'R'}  
    
    # We'll build Takahashi's sequence in a greedy manner:
    # - Try to pick the winning move unless it's the same as the previous move.
    # - If it matches the previous move, pick the draw move (same as Aoki's).
    # This ensures Takahashi never loses (only picks winning or draw),
    # and consecutive moves differ.
    
    takahashi_moves = []
    wins = 0
    
    for i in range(N):
        w = beat[S[i]]         # Winning move against Aoki's move
        if i == 0:             # First move: always choose winning
            takahashi_moves.append(w)
            wins += 1
        else:
            if w != takahashi_moves[-1]:
                takahashi_moves.append(w)
                wins += 1
            else:
                takahashi_moves.append(S[i])  # Draw move
    
    print(wins)

# Do not forget to call main()
if __name__ == "__main__":
    main()