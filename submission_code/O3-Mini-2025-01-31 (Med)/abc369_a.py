def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    A = int(input_data[0])
    B = int(input_data[1])
    
    # The key observation is that the three numbers A, B, and x
    # can form an arithmetic sequence if one of them is the average of the other two.
    # There are three potential cases:
    # 1. A is the average: 2A = x + B, so x = 2A - B.
    # 2. B is the average: 2B = A + x, so x = 2B - A.
    # 3. x is the average: 2x = A + B, so x = (A + B) / 2 (only valid if A+B is even).
    
    candidates = set()
    candidates.add(2 * A - B)
    candidates.add(2 * B - A)
    if (A + B) % 2 == 0:
        candidates.add((A + B) // 2)
    
    print(len(candidates))
    
if __name__ == '__main__':
    main()