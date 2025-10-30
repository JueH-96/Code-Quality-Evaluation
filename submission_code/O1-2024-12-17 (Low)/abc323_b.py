def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    results = data[1:]
    
    # Count the wins for each player i (1-indexed)
    win_counts = [0] * N
    for i in range(N):
        row = results[i]
        # Count 'o' in row i
        win_counts[i] = row.count('o')
    
    # Prepare a list of (player_number, win_count) 
    # Then sort by descending win_count, then ascending player_number
    players = [(i+1, win_counts[i]) for i in range(N)]
    players.sort(key=lambda x: (-x[1], x[0]))
    
    # Output the sorted order of player numbers
    output = [str(p[0]) for p in players]
    print(" ".join(output))

# do not forget to call main
if __name__ == "__main__":
    main()