def main():
    import sys
    data_input = sys.stdin.read().splitlines()
    if not data_input:
        print(0)
        return
        
    n = int(data_input[0].strip())
    index = 1
    players = []
    
    for i in range(n):
        c_i = int(data_input[index]); index += 1
        bets_line = data_input[index].split()
        index += 1
        bets_set = set(map(int, bets_line))
        players.append((c_i, bets_set))
        
    X = int(data_input[index].strip())
    
    min_bets = 1000
    winners = []
    
    for i in range(n):
        c_i, bets = players[i]
        if X in bets:
            if c_i < min_bets:
                min_bets = c_i
                winners = [i+1]
            elif c_i == min_bets:
                winners.append(i+1)
                
    if not winners:
        print(0)
    else:
        print(len(winners))
        print(" ".join(map(str, winners)))

if __name__ == "__main__":
    main()