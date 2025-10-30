def main():
    N = int(input().strip())
    for r in range(N):
        row = []
        for c in range(N):
            # distance to nearest border
            d = min(r, c, N - 1 - r, N - 1 - c)
            # layer number = d+1
            if (d + 1) % 2 == 1:
                row.append('#')
            else:
                row.append('.')
        print(''.join(row))

if __name__ == "__main__":
    main()