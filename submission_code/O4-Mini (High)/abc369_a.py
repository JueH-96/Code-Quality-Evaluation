def main():
    import sys
    data = sys.stdin.read().strip().split()
    A = int(data[0])
    B = int(data[1])
    # Possible x values come from placing each of A, B, x in the middle of the AP:
    # 1) x in middle: 2*x = A+B  => x = (A+B)/2  (must be integer)
    # 2) A in middle: 2*A = B+x  => x = 2*A - B
    # 3) B in middle: 2*B = A+x  => x = 2*B - A
    s = set()
    s.add(2*A - B)
    s.add(2*B - A)
    if (A + B) % 2 == 0:
        s.add((A + B) // 2)
    print(len(s))

if __name__ == "__main__":
    main()