def main():
    import sys
    H = int(sys.stdin.readline())
    height = 0
    day = 0
    # Each morning on day `day` the height is `height`.
    # We apply the nightly growth before the next morning.
    while True:
        day += 1
        height = 2 * height + 1  # growth at night i adds 2^(i-1), leading to h = 2^day - 1
        if height > H:
            print(day)
            return

if __name__ == "__main__":
    main()