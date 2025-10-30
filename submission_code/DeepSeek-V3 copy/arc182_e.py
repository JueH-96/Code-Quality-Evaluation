def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    C = int(data[2])
    K = int(data[3])
    A = list(map(int, data[4:4+N]))
    
    if C == 0:
        min_val = min(A)
        print(min_val * K)
        return
    
    # Compute the minimum for each k in [0, K-1]
    # Since Ck mod M can be large, we need to find the pattern or cycle
    # The sequence Ck mod M repeats every M / gcd(C, M) steps
    gcd_cm = gcd(C, M)
    cycle_length = M // gcd_cm
    
    # We can compute the minimum for each position in the cycle
    # and then sum over the full cycles and the remaining steps
    
    # Precompute the minimum for each position in the cycle
    min_in_cycle = [float('inf')] * cycle_length
    for k in range(cycle_length):
        current_min = float('inf')
        for a in A:
            val = (C * k + a) % M
            if val < current_min:
                current_min = val
        min_in_cycle[k] = current_min
    
    # Calculate the number of full cycles and the remaining steps
    full_cycles = K // cycle_length
    remaining = K % cycle_length
    
    # Sum the full cycles
    sum_full_cycles = full_cycles * sum(min_in_cycle)
    
    # Sum the remaining steps
    sum_remaining = sum(min_in_cycle[:remaining])
    
    total = sum_full_cycles + sum_remaining
    print(total)

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

if __name__ == "__main__":
    main()