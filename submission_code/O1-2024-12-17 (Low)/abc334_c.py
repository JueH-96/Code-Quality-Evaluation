def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    K = int(data[1])
    lost = set(map(int, data[2:]))

    # Prepare a list that contains each color the correct number of times:
    #  - If the color is lost, we have 1 sock of that color
    #  - Otherwise, we have 2 socks of that color
    socks = []
    for color in range(1, N + 1):
        if color in lost:
            socks.append(color)
        else:
            socks.append(color)
            socks.append(color)

    # Sort the socks by color
    socks.sort()

    # We'll pair every two adjacent socks, ignoring one leftover if the total is odd
    pairs_count = (2 * N - K) // 2
    total_weirdness = 0
    idx = 0
    for _ in range(pairs_count):
        total_weirdness += abs(socks[idx] - socks[idx + 1])
        idx += 2

    print(total_weirdness)

# Do not forget to call main() at the end
if __name__ == "__main__":
    main()