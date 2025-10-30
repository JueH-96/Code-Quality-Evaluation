import sys

def main():
    data = sys.stdin.read().split()
    idx = 0
    N = int(data[idx])
    M = int(data[idx + 1])
    idx += 2
    C = data[idx:idx + N]
    idx += N
    D = data[idx:idx + M]
    idx += M
    P = list(map(int, data[idx:idx + M + 1]))
    idx += M + 1

    color_price = {d: P[i + 1] for i, d in enumerate(D)}
    total = 0
    for color in C:
        if color in color_price:
            total += color_price[color]
        else:
            total += P[0]
    print(total)

if __name__ == "__main__":
    main()