def main():
    import sys
    input = sys.stdin.readline
    
    N = int(input().strip())
    results = [input().strip() for _ in range(N)]
    
    # Calculate wins for each player
    wins = []
    for i in range(N):
        win_count = results[i].count('o')
        wins.append((win_count, i + 1))  # store (wins, player number)
    
    # sort based on wins in descending order and player number in ascending order
    wins.sort(key=lambda x: (-x[0], x[1]))
    
    # Extract sorted player numbers
    sorted_players = [str(player) for _, player in wins]
    print(" ".join(sorted_players))

if __name__ == '__main__':
    main()