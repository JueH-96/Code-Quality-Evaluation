# YOUR CODE HERE
import sys

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    index = 1
    people = []
    for _ in range(N):
        C = int(data[index])
        index += 1
        bets = list(map(int, data[index:index + C]))
        index += C
        people.append((C, bets))
    
    X = int(data[-1])
    
    min_bets = float('inf')
    winners = []
    
    for i, (C, bets) in enumerate(people, start=1):
        if X in bets:
            if C < min_bets:
                min_bets = C
                winners = [i]
            elif C == min_bets:
                winners.append(i)
    
    print(len(winners))
    if winners:
        print(' '.join(map(str, winners)))

if __name__ == "__main__":
    main()