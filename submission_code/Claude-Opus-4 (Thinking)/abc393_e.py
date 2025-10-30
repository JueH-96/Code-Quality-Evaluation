# YOUR CODE HERE
N, K = map(int, input().split())
A = list(map(int, input().split()))

max_val = max(A)

# For each divisor d, count how many elements are divisible by d
count = [0] * (max_val + 1)

for a in A:
    # Find all divisors of a
    d = 1
    while d * d <= a:
        if a % d == 0:
            count[d] += 1
            if d != a // d:
                count[a // d] += 1
        d += 1

# For each position i
for i in range(N):
    max_gcd = 0
    
    # Find all divisors of A[i]
    d = 1
    while d * d <= A[i]:
        if A[i] % d == 0:
            # Check if we can choose K elements with GCD = d
            if count[d] >= K:
                max_gcd = max(max_gcd, d)
            
            # Also check A[i] // d
            other_d = A[i] // d
            if d != other_d and count[other_d] >= K:
                max_gcd = max(max_gcd, other_d)
        d += 1
    
    print(max_gcd)