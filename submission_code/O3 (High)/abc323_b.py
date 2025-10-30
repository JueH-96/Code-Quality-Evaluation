def main() -> None:
    import sys

    input = sys.stdin.readline

    # Read number of players
    N = int(input().strip())

    wins = []  # list of tuples: (number_of_wins, player_id)

    # Read results and compute wins for each player
    for player_id in range(1, N + 1):
        s = input().strip()
        win_count = s.count('o')  # each 'o' in row means this player won
        wins.append((win_count, player_id))

    # Sort: more wins first, then smaller player number
    ranking = sorted(wins, key=lambda x: (-x[0], x[1]))

    # Output player numbers in order
    print(' '.join(str(player_id) for _, player_id in ranking))


if __name__ == "__main__":
    main()