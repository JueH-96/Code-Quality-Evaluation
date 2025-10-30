# YOUR CODE HERE
import sys

def main():
    N, M, C, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    
    # Precompute the minimum for each possible (Ck) mod M
    # Since (Ck) mod M cycles every M / gcd(C, M) steps
    # We need to find the period of the sequence (Ck) mod M
    
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    g = gcd(C, M)
    period = M // g
    
    # Precompute the minimum for each residue in the period
    # We need to find for each residue r in 0..period-1, the minimum (r + A_i) mod M
    
    # Create a list to store the minimum for each residue
    min_for_residue = [float('inf')] * period
    
    for r in range(period):
        current_min = float('inf')
        for a in A:
            val = (r + a) % M
            if val < current_min:
                current_min = val
        min_for_residue[r] = current_min
    
    # Now, we need to find how many full periods are in K, and the remaining steps
    full_periods = K // period
    remaining = K % period
    
    # Calculate the sum for full periods
    sum_full = full_periods * sum(min_for_residue)
    
    # Calculate the sum for the remaining steps
    sum_remaining = 0
    for i in range(remaining):
        sum_remaining += min_for_residue[i % period]
    
    total = sum_full + sum_remaining
    print(total)

if __name__ == "__main__":
    main()