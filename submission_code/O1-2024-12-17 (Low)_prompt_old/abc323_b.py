def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    S = data[1:]  # the next N lines
    
    # Calculate number of wins for each player
    wins = []
    for i in range(N):
        count_wins = S[i].count('o')
        wins.append((count_wins, i+1))  # store (wins, player_number)
    
    # Sort by descending number of wins, then by ascending player number
    wins.sort(key=lambda x: (-x[0], x[1]))
    
    # Print player numbers in the required order
    print(" ".join(str(player_num) for _, player_num in wins))

def main():
    solve()

if __name__ == "__main__":
    main()