def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    takahashi_score = 0
    aoki_score = 0
    for _ in range(n):
        x = int(next(it))
        y = int(next(it))
        takahashi_score += x
        aoki_score += y

    if takahashi_score > aoki_score:
        print("Takahashi")
    elif takahashi_score < aoki_score:
        print("Aoki")
    else:
        print("Draw")

if __name__ == "__main__":
    main()