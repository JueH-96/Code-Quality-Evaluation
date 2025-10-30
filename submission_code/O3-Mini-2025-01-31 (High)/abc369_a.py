def main():
    import sys
    data = sys.stdin.read().split()
    A = int(data[0])
    B = int(data[1])
    
    # To form an arithmetic sequence from A, B, and x,
    # one of these must be the arithmetic mean of the other two.
    # That gives us the conditions:
    # 1. A is the average: A = (B + x) / 2  ->  x = 2*A - B
    # 2. B is the average: B = (A + x) / 2  ->  x = 2*B - A
    # 3. x is the average: x = (A + B) / 2  (only valid if A+B is even)
    
    candidates = set()
    candidates.add(2 * A - B)
    candidates.add(2 * B - A)
    
    if (A + B) % 2 == 0:
        candidates.add((A + B) // 2)
    
    print(len(candidates))

if __name__ == '__main__':
    main()