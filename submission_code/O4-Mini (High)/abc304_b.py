def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    s = str(N)
    # If N has at most 3 digits, print it unchanged
    if len(s) <= 3:
        print(N)
    else:
        # For d-digit N (d >= 4), zero out the last (d-3) digits:
        d = len(s)
        power = 10 ** (d - 3)
        print((N // power) * power)

if __name__ == "__main__":
    main()