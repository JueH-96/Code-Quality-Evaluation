def main():
    import sys
    H = int(sys.stdin.read().strip())
    
    day = 0
    # We'll keep doubling until 2^day - 1 > H
    # Because the plant's height in the morning of day i is 2^i - 1.
    while True:
        day += 1
        if (1 << day) - 1 > H:  # (1 << day) is 2^day
            print(day)
            break

# Call main function
if __name__ == "__main__":
    main()