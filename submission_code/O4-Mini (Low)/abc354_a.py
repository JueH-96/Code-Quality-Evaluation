def main():
    import sys
    H = int(sys.stdin.readline())
    height = 0
    day = 0
    # Each loop represents the night of day 'day', adding 2^day cm
    # The new height is then observed the next morning (day+1).
    while True:
        height += 1 << day  # add 2^day
        day += 1
        if height > H:
            print(day)
            return

if __name__ == "__main__":
    main()