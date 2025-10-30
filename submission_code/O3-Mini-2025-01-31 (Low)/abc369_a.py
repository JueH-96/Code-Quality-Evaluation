def main():
    import sys
    input_data = sys.stdin.read().split()
    A = int(input_data[0])
    B = int(input_data[1])
    
    candidates = set()
    
    # Case 1: Arrange as [A, x, B] or [B, x, A] where x is the mid of A and B
    # Thus, x must be the average of A and B (if A+B is even)
    if (A+B) % 2 == 0:
        x3 = (A+B) // 2
        candidates.add(x3)
    
    # Case 2: x as one of the ends such that A becomes the middle: when arranging [x, A, B] or [B, A, x],
    # We require A = (x + B)/2, thus x = 2*A - B.
    x1 = 2*A - B
    candidates.add(x1)
    
    # Case 3: x as one of the ends such that B becomes the middle: when arranging [x, B, A] or [A, B, x],
    # We require B = (x + A)/2, thus x = 2*B - A.
    x2 = 2*B - A
    candidates.add(x2)
    
    print(len(candidates))

if __name__ == '__main__':
    main()