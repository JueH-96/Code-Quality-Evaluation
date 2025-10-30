import sys

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    results = data[1:]

    wins = [0] * N

    for i in range(N):
        for j in range(N):
            if results[i][j] == 'o':
                wins[i] += 1

    ranked_players = sorted(range(1, N + 1), key=lambda x: (-wins[x - 1], x))

    print(' '.join(map(str, ranked_players)))

if __name__ == "__main__":
    main()