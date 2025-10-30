import sys
import math

def main():
    # Read input
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx]); idx +=1
    M = int(input[idx]); idx +=1
    C = int(input[idx]); idx +=1
    K = int(input[idx]); idx +=1
    A = list(map(int, input[idx:idx+N]))
    
    # Compute the period T
    d = math.gcd(C, M)
    T = M // d
    
    # Precompute min_list
    min_list = []
    for k in range(T):
        current_min = M  # Initialize with a value larger than any possible mod result
        for a in A:
            val = (C * k + a) % M
            if val < current_min:
                current_min = val
                if current_min == 0:  # Can't get lower than 0
                    break
        min_list.append(current_min)
    
    # Calculate the sum
    sum_cycle = sum(min_list)
    full_cycles = K // T
    rem = K % T
    sum_rem = sum(min_list[:rem])
    total = sum_cycle * full_cycles + sum_rem
    
    print(total)

if __name__ == '__main__':
    main()