def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    # The next N entries in data are the strings S_1, ..., S_N
    results = data[1:]  # This will be a list of N strings

    # Calculate number of wins for each player
    wins = []
    for i in range(N):
        # Count 'o' in the i-th string
        win_count = results[i].count('o')
        wins.append((win_count, i+1))

    # Sort by wins (descending), then by player number (ascending)
    wins.sort(key=lambda x: (-x[0], x[1]))

    # Print the player numbers in the required order
    print(" ".join(str(player) for _, player in wins))

# Do not forget to call main()
main()