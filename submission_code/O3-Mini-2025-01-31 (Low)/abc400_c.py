def main():
    import sys, math
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    N = int(input_data[0])
    
    count = 0
    a = 1
    # For each positive integer exponent a, we require 2^a * b^2 <= N.
    # b^2 must then satisfy: b^2 <= N / (2^a)
    # b can be any positive integer with b^2 <= floor(N / (2^a))
    # Loop until 2^a exceeds N.
    while True:
        power = 1 << a  # 2^a
        if power > N:
            break
        limit = N // power
        # using math.isqrt to compute floor(sqrt(limit))
        b_max = math.isqrt(limit)
        
        count += b_max
        a += 1
    
    sys.stdout.write(str(count))
    
if __name__ == '__main__':
    main()