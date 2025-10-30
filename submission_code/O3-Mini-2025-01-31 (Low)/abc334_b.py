def main():
    import sys
    import math
    
    input_line = sys.stdin.read().strip().split()
    if not input_line:
        return
    A, M, L, R = map(int, input_line)
    
    # Calculate smallest integer k (k_min) such that A + kM >= L
    # Solve k >= (L - A) / M, so k_min = ceil((L - A) / M)
    k_min = math.ceil((L - A) / M)
    
    # Calculate largest integer k (k_max) such that A + kM <= R
    # Solve k <= (R - A) / M, so k_max = floor((R - A) / M)
    k_max = math.floor((R - A) / M)
    
    # Count is number of integers from k_min to k_max, if any
    if k_max < k_min:
        count = 0
    else:
        count = k_max - k_min + 1
    
    print(count)

if __name__ == '__main__':
    main()