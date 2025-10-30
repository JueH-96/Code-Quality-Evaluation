def main():
    import sys
    data = sys.stdin.read().strip().split()
    A, B = map(int, data)
    
    xs = set()
    # Case 1: x is the middle term, so 2x = A + B
    if (A + B) % 2 == 0:
        xs.add((A + B) // 2)
    # Case 2: A is the middle term, so 2A = B + x  => x = 2A - B
    xs.add(2 * A - B)
    # Case 3: B is the middle term, so 2B = A + x  => x = 2B - A
    xs.add(2 * B - A)
    
    print(len(xs))

if __name__ == "__main__":
    main()