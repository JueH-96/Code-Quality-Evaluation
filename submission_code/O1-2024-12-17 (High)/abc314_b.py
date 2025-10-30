def main():
    import sys

    # Read the number of people
    N = int(sys.stdin.readline().strip())

    # Lists to store each person's bet count and the set of bets
    C = []
    bets = []

    # Read bet information for each person
    for _ in range(N):
        c_i = int(sys.stdin.readline().strip())
        C.append(c_i)
        arr = list(map(int, sys.stdin.readline().strip().split()))
        bets.append(set(arr))

    # Read the outcome of the spin
    X = int(sys.stdin.readline().strip())

    # Collect indices of people who bet on X
    bet_on_X = [i for i in range(N) if X in bets[i]]

    # If no one bet on X, print 0 and return
    if not bet_on_X:
        print(0)
        return

    # Find the minimum number of bets among those who did bet on X
    min_bets = min(C[i] for i in bet_on_X)

    # Collect all people who bet on X and whose bet count is min_bets
    result = [i+1 for i in bet_on_X if C[i] == min_bets]

    # Print the number of such people
    print(len(result))
    # If there are any, print their indices in ascending order
    if result:
        print(*result)

# Do not forget to call main() at the end
if __name__ == "__main__":
    main()