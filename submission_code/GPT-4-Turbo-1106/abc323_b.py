def main():
    N = int(input().strip())
    results = [input().strip() for _ in range(N)]

    # Calculate the number of wins for each player
    wins = [0] * N
    for i in range(N):
        for j in range(N):
            if results[i][j] == 'o':
                wins[i] += 1

    # Sort the players by the number of wins, then by player number
    sorted_players = sorted(range(1, N + 1), key=lambda x: (-wins[x - 1], x))

    # Output the sorted player numbers
    print(' '.join(map(str, sorted_players)))

if __name__ == "__main__":
    main()