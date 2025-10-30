def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    A = int(data[0])
    B = int(data[1])
    
    if A > B:
        A, B = B, A
    
    # A is guaranteed to be less than or equal to B now
    count = 0
    # Case 1: x is the middle element (A, x, B)
    x1 = 2 * B - A
    count += 1
    
    # Case 2: x is the middle element (B, x, A)
    x2 = 2 * A - B
    count += 1
    
    # Case 3: x is either the first or last element (x, A, B) or (A, B, x)
    x3 = A + (B - A) * 2
    x4 = A - (B - A)
    
    # We need to count unique x's
    unique_xs = {x1, x2, x3, x4}
    print(len(unique_xs))

if __name__ == "__main__":
    main()