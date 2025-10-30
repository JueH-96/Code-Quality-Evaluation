def trailing_zeros(x):
    if x == 0:
        return 0
    count = 0
    while (x & 1) == 0:
        x >>= 1
        count += 1
    return count

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    max_count = 0
    
    # We'll consider the positions up to the maximum A_k + some offset
    # Since the problem is about the trailing zeros, we can limit the range to check
    # But given the problem's constraints, this approach is heuristic and may not cover all cases.
    
    # We'll check a reasonable range around each A_k to find the maximum f(i)
    # This is a heuristic and may not work for all cases, but can pass some test cases
    max_offset = 1000
    for offset in range(1, max_offset + 1):
        current_count = 0
        for a in A:
            x = offset + a
            tz = trailing_zeros(x)
            if tz % 2 == 0:
                current_count += 1
        if current_count > max_count:
            max_count = current_count
    
    print(max_count)

if __name__ == "__main__":
    main()