def main():
    import sys
    data = sys.stdin.read().strip().split()
    
    N = int(data[0])
    index = 1
    
    bets = []
    for i in range(N):
        Ci = int(data[index])
        index += 1
        outcomes = set(map(int, data[index:index+Ci]))
        index += Ci
        bets.append((Ci, outcomes))
    
    X = int(data[index])  # outcome of the spin
    
    # Find all players who bet on X
    players_who_bet_on_X = [(i+1, bets[i][0]) for i in range(N) if X in bets[i][1]]
    
    if not players_who_bet_on_X:
        # No one bet on X
        print(0)
        return

    # Find the minimum number of bets among the players who bet on X
    min_bets = min(players_who_bet_on_X, key=lambda x: x[1])[1]

    # Filter players who bet on X and have that minimum number of bets
    result = [player for (player, count) in players_who_bet_on_X if count == min_bets]

    # Print the results
    print(len(result))
    if len(result) > 0:
        print(" ".join(map(str, sorted(result))))

# Do not forget to call main()
main()