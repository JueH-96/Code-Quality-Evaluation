import math

def main():
    n = int(input())
    a = list(map(int, input().split()))
    mod = 998244353
    
    # Precompute powers of 2
    powers_of_2 = [0] * n
    powers_of_2[0] = 1
    for i in range(1, n):
        powers_of_2[i] = (powers_of_2[i - 1] * 2) % mod
    
    total_sum = 0
    for m in range(1, n + 1):
        if m > 1:
            # Double the previous sum for new subsequences
            total_sum = (total_sum * 2) % mod
            
            # Add new contributions from pairs where A_m-1 is the right element
            for i in range(m - 1):
                # For each pair (A_i, A_m-1), calculate how many subsequences have them adjacent
                # This happens when all elements between them are not included
                # The contribution is: gcd(A_i, A_m-1) * 2^i
                contribution = (math.gcd(a[i], a[m - 1]) * powers_of_2[i]) % mod
                total_sum = (total_sum + contribution) % mod
        
        print(total_sum)

if __name__ == "__main__":
    main()