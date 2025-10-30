def main():
    import sys
    input = sys.stdin.readline

    N = int(input())
    takahashi_score = 0
    aoki_score = 0

    for _ in range(N):
        x, y = map(int, input().split())
        takahashi_score += x
        aoki_score += y

    if takahashi_score > aoki_score:
        print("Takahashi")
    elif aoki_score > takahashi_score:
        print("Aoki")
    else:
        print("Draw")

if __name__ == "__main__":
    main()