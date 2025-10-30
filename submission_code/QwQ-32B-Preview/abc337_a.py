def main():
    import sys
    data = sys.stdin.read().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    total_takahashi = 0
    total_aoki = 0
    for _ in range(N):
        X = int(data[idx])
        idx += 1
        Y = int(data[idx])
        idx += 1
        total_takahashi += X
        total_aoki += Y
    if total_takahashi > total_aoki:
        print("Takahashi")
    elif total_aoki > total_takahashi:
        print("Aoki")
    else:
        print("Draw")

if __name__ == "__main__":
    main()