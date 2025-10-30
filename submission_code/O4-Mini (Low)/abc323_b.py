def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    S = data[1:]
    # Count wins for each player i (0-based index)
    wins = [s.count('o') for s in S]
    # Create list of (wins, -player_index) for sorting; we use -index so smaller index ranks higher on tie
    players = list(range(1, N+1))
    # Sort by descending wins, then ascending player number
    players.sort(key=lambda i: (-wins[i-1], i))
    # Print result
    print(" ".join(map(str, players)))

if __name__ == "__main__":
    main()