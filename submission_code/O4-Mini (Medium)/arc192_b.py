def main():
    import sys
    input = sys.stdin.readline
    N = int(input())
    _ = input()  # we don't actually need the A_i values
    # According to the analysis, the first player (Fennec) wins
    # iff N is odd, otherwise Snuke wins.
    if N % 2 == 1:
        print("Fennec")
    else:
        print("Snuke")

if __name__ == "__main__":
    main()