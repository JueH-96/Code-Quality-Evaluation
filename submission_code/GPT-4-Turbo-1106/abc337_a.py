def main():
    N = int(input().strip())
    takahashi_score = 0
    aoki_score = 0

    for _ in range(N):
        X, Y = map(int, input().split())
        takahashi_score += X
        aoki_score += Y

    if takahashi_score > aoki_score:
        print("Takahashi")
    elif takahashi_score < aoki_score:
        print("Aoki")
    else:
        print("Draw")

if __name__ == "__main__":
    main()