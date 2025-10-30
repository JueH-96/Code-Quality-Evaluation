def solve():
    a, b, c, d = map(int, input().split())

    black_area = 0
    for i in range(a, c):
        for j in range(b, d):
            c1 = i % 2 == 0
            c2 = (j // 2) % 2 == 0
            c3 = ((i + j) // 2) % 2 == 0

            if (c1 and c2 and c3) or \
               (c1 and not c2 and not c3) or \
               (not c1 and c2 and not c3) or \
               (not c1 and not c2 and c3):
                black_area += 1

    print(2 * black_area)

if __name__ == "__main__":
    solve()