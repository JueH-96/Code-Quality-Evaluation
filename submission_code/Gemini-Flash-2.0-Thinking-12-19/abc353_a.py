def solve():
    n = int(input())
    h = list(map(int, input().split()))

    first_height = h[0]
    leftmost_taller_position = -1

    for i in range(1, n):
        if h[i] > first_height:
            leftmost_taller_position = i + 1
            break

    print(leftmost_taller_position)

if __name__ == "__main__":
    solve()