def get_divisors(n):
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    return sorted(divisors, reverse=True)

N, K = map(int, input().split())
A = list(map(int, input().split()))

# Count how many elements are divisible by each number
max_val = max(A)
count = [0] * (max_val + 1)

for num in A:
    for d in range(1, int(num**0.5) + 1):
        if num % d == 0:
            count[d] += 1
            if d != num // d:
                count[num // d] += 1

# For each position i
for i in range(N):
    # Get divisors of A[i] in decreasing order
    divisors = get_divisors(A[i])
    
    # Find the largest divisor d such that at least K elements are divisible by d
    for d in divisors:
        if count[d] >= K:
            print(d)
            break