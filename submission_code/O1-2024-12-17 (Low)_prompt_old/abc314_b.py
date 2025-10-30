def solve():
    import sys
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    
    bets_counts = []
    all_bets = []
    idx = 1
    
    # Read input
    for _ in range(N):
        C_i = int(data[idx])
        idx += 1
        bets = list(map(int, data[idx:idx+C_i]))
        idx += C_i
        
        bets_counts.append(C_i)
        all_bets.append(set(bets))
    
    X = int(data[idx])
    
    # Find those who bet on X
    people_bet_on_X = []
    for i in range(N):
        if X in all_bets[i]:
            people_bet_on_X.append(i + 1)  # +1 because persons numbered from 1
    
    # If nobody bet on X, print 0 and return
    if not people_bet_on_X:
        print(0)
        print()  # print empty line
        return
    
    # Among those who bet on X, find the fewest bets
    min_bets = min(bets_counts[i-1] for i in people_bet_on_X)
    
    # Collect all who have C_i = min_bets
    answer = [i for i in people_bet_on_X if bets_counts[i-1] == min_bets]
    
    # Print results
    print(len(answer))
    if answer:
        print(" ".join(map(str, answer)))

def main():
    solve()

if __name__ == "__main__":
    main()