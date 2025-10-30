import sys

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    idx = 1
    bets = []
    for i in range(N):
        C_i = int(data[idx])
        idx += 1
        A_i = list(map(int, data[idx:idx+C_i]))
        idx += C_i
        bets.append(A_i)
    X = int(data[-1])
    
    min_bets = float('inf')
    winners = []
    for i, bet in enumerate(bets):
        if X in bet:
            if len(bet) < min_bets:
                min_bets = len(bet)
                winners = [i+1]
            elif len(bet) == min_bets:
                winners.append(i+1)
    
    print(len(winners))
    print(*winners)

if __name__ == "__main__":
    main()