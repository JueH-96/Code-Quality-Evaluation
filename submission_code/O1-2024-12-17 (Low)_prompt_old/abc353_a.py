def solve():
    import sys

    data = sys.stdin.read().strip().split()
    N = int(data[0])
    heights = list(map(int, data[1:]))

    first_height = heights[0]
    answer = -1

    for i in range(1, N):
        if heights[i] > first_height:
            answer = i + 1
            break

    print(answer)

def main():
    solve()

if __name__ == "__main__":
    main()