def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    n = int(data[0])
    # Isolate the lowest set bit: (n & -n) == 2^k where k is the count of trailing zeros
    # bit_length() of 2^k is k+1, so subtract 1.
    ctz = (n & -n).bit_length() - 1
    print(ctz)

if __name__ == "__main__":
    main()