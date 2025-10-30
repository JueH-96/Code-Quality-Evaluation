def main():
    import sys
    data = sys.stdin.read().strip()
    if not data:
        return
    N = int(data)
    # Convert (N-1) to base-5, then map digits 0->0,1->2,2->4,3->6,4->8
    M = N - 1
    # Special case for the very first good integer
    if M == 0:
        print(0)
        return
    # Build base-5 digits (least significant first)
    digits = []
    while M > 0:
        digits.append(M % 5)
        M //= 5
    # Reverse to get most significant first
    digits.reverse()
    # Map each base-5 digit to its even-digit equivalent
    # 0->0, 1->2, 2->4, 3->6, 4->8
    res = ''.join(str(d * 2) for d in digits)
    print(res)

if __name__ == "__main__":
    main()