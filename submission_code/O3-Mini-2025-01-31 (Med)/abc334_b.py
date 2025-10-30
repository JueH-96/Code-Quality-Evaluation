def main():
    import sys
    import math
    # Read input values
    data = sys.stdin.read().split()
    A = int(data[0])
    M = int(data[1])
    L = int(data[2])
    R = int(data[3])
    
    # Determine the smallest integer k such that A + k*M >= L.
    k_min = math.ceil((L - A) / M)
    # Determine the largest integer k such that A + k*M <= R.
    k_max = math.floor((R - A) / M)
    
    # Calculate the number of Christmas trees in the interval.
    count = max(0, k_max - k_min + 1)
    
    print(count)

if __name__ == '__main__':
    main()