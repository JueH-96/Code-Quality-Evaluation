def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    N = int(data[index])
    index += 1
    
    bets = []
    
    for _ in range(N):
        C_i = int(data[index])
        index += 1
        A_i = list(map(int, data[index:index + C_i]))
        index += C_i
        bets.append(A_i)
    
    X = int(data[index])
    
    # Find all persons who bet on X and track their number of bets
    persons_who_bet_on_X = []
    for i in range(N):
        if X in bets[i]:
            persons_who_bet_on_X.append((i + 1, len(bets[i])))
    
    # Find the minimum number of bets among those who bet on X
    if not persons_who_bet_on_X:
        print(0)
        print()
        return
    
    min_bets = min(count for _, count in persons_who_bet_on_X)
    
    # Collect all persons who bet on X and have the minimum number of bets
    result = [person for person, count in persons_who_bet_on_X if count == min_bets]
    
    # Print results
    print(len(result))
    if result:
        print(" ".join(map(str, sorted(result))))
    else:
        print()

if __name__ == "__main__":
    main()