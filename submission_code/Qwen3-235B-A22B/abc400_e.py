import sys
import math
import bisect

def main():
    input = sys.stdin.read().split()
    Q = int(input[0])
    queries = list(map(int, input[1:Q+1]))
    
    max_x = 10**6  # since sqrt(1e12) is 1e6
    omega = [0] * (max_x + 1)
    
    # Compute the number of distinct prime factors using sieve
    for i in range(2, max_x + 1):
        if omega[i] == 0:  # i is a prime
            for j in range(i, max_x + 1, i):
                omega[j] += 1
    
    # Collect all valid X where omega[X] == 2
    valid_X = []
    for x in range(2, max_x + 1):
        if omega[x] == 2:
            valid_X.append(x)
    
    # Process each query
    output = []
    for a in queries:
        s = math.isqrt(a)
        # Find the largest X <= s using binary search
        idx = bisect.bisect_right(valid_X, s) - 1
        x = valid_X[idx]
        output.append(str(x * x))
    
    print('
'.join(output))

if __name__ == '__main__':
    main()