def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return

    # Read N (total keys), M (number of tests), and K (minimum real keys needed to open door)
    N = int(data[0])
    M = int(data[1])
    K = int(data[2])
    pos = 3

    # Process each test: build a bitmask for keys used in the test and record the result.
    tests = []
    for _ in range(M):
        C = int(data[pos])
        pos += 1
        keys = list(map(int, data[pos: pos + C]))
        pos += C
        result = data[pos]
        pos += 1
        
        # Build a bitmask (bit i corresponds to key i+1)
        mask = 0
        for key in keys:
            mask |= 1 << (key - 1)
        tests.append((mask, result))
    
    count_valid = 0
    # There are 2^N possible assignments for real/dummy keys.
    for assignment in range(1 << N):
        valid = True
        for tmask, res in tests:
            # Count the number of real keys in the current test assignment.
            real_count = (assignment & tmask).bit_count()
            if res == 'o':
                if real_count < K:
                    valid = False
                    break
            else:  # res == 'x'
                if real_count >= K:
                    valid = False
                    break
        if valid:
            count_valid += 1

    print(count_valid)

if __name__ == '__main__':
    main()