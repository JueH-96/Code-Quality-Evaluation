n, k = map(int, input().split())
A = list(map(int, input().split()))

max_val = max(A)

# Count frequencies of each element value
freq = [0] * (max_val + 1)
for a in A:
    freq[a] += 1

# For each possible divisor d, count how many elements are divisible by d
divisible_count = [0] * (max_val + 1)
for d in range(1, max_val + 1):
    for multiple in range(d, max_val + 1, d):
        divisible_count[d] += freq[multiple]

# For each A_i, find the maximum GCD
for i in range(n):
    a_i = A[i]
    
    # Find all divisors of A_i
    divisors = []
    for d in range(1, int(a_i**0.5) + 1):
        if a_i % d == 0:
            divisors.append(d)
            if d != a_i // d:
                divisors.append(a_i // d)
    
    # Sort divisors in descending order and find the first valid one
    divisors.sort(reverse=True)
    
    result = 1  # fallback
    for d in divisors:
        if divisible_count[d] >= k:
            result = d
            break
    
    print(result)