import sys
input = sys.stdin.read

def main():
    data = input().split()
    index = 0

    N = int(data[index])
    M = int(data[index + 1])
    L = int(data[index + 2])
    index += 3

    a = [int(data[index + i]) for i in range(N)]
    index += N

    b = [int(data[index + i]) for i in range(M)]
    index += M

    not_offered = set((int(data[index + 2 * i]), int(data[index + 2 * i + 1])) for i in range(L))
    index += 2 * L

    max_price = 0

    for i in range(1, N + 1):
        for j in range(1, M + 1):
            if (i, j) not in not_offered:
                price = a[i - 1] + b[j - 1]
                if price > max_price:
                    max_price = price

    print(max_price)

if __name__ == "__main__":
    main()