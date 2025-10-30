def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    # data will contain [N, C1, A11, A12, ..., A1C1, C2, A21, A22, ...]
    
    index = 0
    N = int(data[index]); index += 1
    
    bets = []  # list of tuples (C_i, set_of_bets)
    for _ in range(N):
        C_i = int(data[index])
        index += 1
        bet_numbers = set(int(x) for x in data[index:index + C_i])
        index += C_i
        bets.append((C_i, bet_numbers))
    
    X = int(data[index])
    
    # Filter who has bet on X
    filtered = [(i+1, bets[i][0]) for i in range(N) if X in bets[i][1]]
    
    if not filtered:
        print(0)
        print()
        return
    
    # find the minimal number of C_i among those who bet on X
    min_bets = min(p[1] for p in filtered)
    
    # gather all i's that have that minimal number of bets
    result = [p[0] for p in filtered if p[1] == min_bets]
    
    # print result
    print(len(result))
    if len(result) > 0:
        print(" ".join(map(str, result)))

# Call main function
if __name__ == "__main__":
    main()