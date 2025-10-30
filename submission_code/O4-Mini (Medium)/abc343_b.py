def main():
    import sys
    input = sys.stdin.readline

    N = int(input().strip())
    for _ in range(N):
        row = list(map(int, input().split()))
        neighbors = [str(i+1) for i, v in enumerate(row) if v == 1]
        print(" ".join(neighbors))

if __name__ == "__main__":
    main()