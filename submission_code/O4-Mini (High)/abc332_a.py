def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    S = int(data[1])
    K = int(data[2])
    total = 0
    idx = 3
    for _ in range(N):
        P = int(data[idx])
        Q = int(data[idx+1])
        total += P * Q
        idx += 2
    if total < S:
        total += K
    print(total)

if __name__ == "__main__":
    main()