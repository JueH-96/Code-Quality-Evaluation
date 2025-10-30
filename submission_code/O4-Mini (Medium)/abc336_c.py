def main():
    import sys
    data = sys.stdin.read().strip()
    if not data:
        return
    N = int(data)
    # We want the N-th good integer (0-based index is N-1),
    # whose base-5 representation digits are mapped to {0,2,4,6,8} by multiplying by 2.
    x = N - 1
    # handle the zero case specially
    if x == 0:
        print(0)
        return
    # collect base-5 digits of x (least significant first)
    digits = []
    while x > 0:
        digits.append(x % 5)
        x //= 5
    # reverse to get most significant first and map each digit to decimal by *2
    digits.reverse()
    # build the output string
    res = ''.join(str(d * 2) for d in digits)
    print(res)

if __name__ == "__main__":
    main()