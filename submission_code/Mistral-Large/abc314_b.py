import sys
input = sys.stdin.read

def main():
    data = input().split()
    index = 0

    N = int(data[index])
    index += 1

    bets = []
    for i in range(N):
        C_i = int(data[index])
        index += 1
        A_i = list(map(int, data[index:index + C_i]))
        index += C_i
        bets.append((C_i, A_i))

    X = int(data[index])

    min_bets = float('inf')
    winners = []

    for i in range(N):
        C_i, A_i = bets[i]
        if X in A_i:
            if C_i < min_bets:
                min_bets = C_i
                winners = [i + 1]
            elif C_i == min_bets:
                winners.append(i + 1)

    print(len(winners))
    if winners:
        print(" ".join(map(str, winners)))

if __name__ == "__main__":
    main()