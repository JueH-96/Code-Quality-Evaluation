def main():
    import sys
    input = sys.stdin.readline

    N = int(input().strip())
    adj = [list(map(int, input().split())) for _ in range(N)]

    for i in range(N):
        neighbors = [str(j+1) for j in range(N) if adj[i][j] == 1]
        if neighbors:
            print(" ".join(neighbors))
        else:
            print()

if __name__ == "__main__":
    main()