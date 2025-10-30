def main():
    import sys
    import math
    input = sys.stdin.readline
    N = int(input())
    
    seen = set()
    
    # For b >= 2. We consider exponents b.
    # The maximum exponent b is such that 2^b <= N (lowest base is 2). So b_max = floor(log2(N)).
    if N < 1:
        print(0)
        return

    # We can include 1 as well: 1 = 1^b is true for any b >= 2. 
    # But it will be added anyway when a=1, so no special handling is needed.
    
    # Compute maximum possible exponent value: because even though when a >= 1, 
    # for b > floor(log2(N)), 1^b will be the only possibility (which is 1).
    max_b = math.floor(math.log(N, 2)) if N > 1 else 2

    for b in range(2, max_b + 1):
        a = 1
        while True:
            power = a**b
            if power > N:
                break
            seen.add(power)
            a += 1

    print(len(seen))

if __name__ == '__main__':
    main()