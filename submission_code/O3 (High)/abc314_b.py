import sys

def main() -> None:
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    
    min_bets = float('inf')
    winners = []

    # To evaluate at the end we need X, but we only know it after reading all players.
    # So first read players into a list.
    players_bets = []
    for _ in range(N):
        C = int(next(it))
        bets = [int(next(it)) for _ in range(C)]
        players_bets.append((C, bets))
    
    X = int(next(it))

    # Determine winners
    for idx, (C, bets) in enumerate(players_bets, start=1):
        if X in bets:
            if C < min_bets:
                min_bets = C
                winners = [idx]
            elif C == min_bets:
                winners.append(idx)

    K = len(winners)
    print(K)
    if K:
        print(*winners)


if __name__ == "__main__":
    main()