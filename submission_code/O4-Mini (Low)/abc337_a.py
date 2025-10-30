def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    tak = 0
    aok = 0
    idx = 1
    for _ in range(N):
        x = int(data[idx]); y = int(data[idx+1])
        idx += 2
        tak += x
        aok += y
    if tak > aok:
        print("Takahashi")
    elif aok > tak:
        print("Aoki")
    else:
        print("Draw")

if __name__ == "__main__":
    main()