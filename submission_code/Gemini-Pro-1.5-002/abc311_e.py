# YOUR CODE HERE
def solve():
    h, w, n = map(int, input().split())
    holes = set()
    for _ in range(n):
        r, c = map(int, input().split())
        holes.add((r - 1, c - 1))

    count = 0
    for i in range(h):
        for j in range(w):
            for size in range(1, min(h - i, w - j) + 1):
                is_holeless = True
                for row in range(i, i + size):
                    for col in range(j, j + size):
                        if (row, col) in holes:
                            is_holeless = False
                            break
                    if not is_holeless:
                        break
                if is_holeless:
                    count += 1

    print(count)

solve()