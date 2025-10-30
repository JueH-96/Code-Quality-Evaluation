# Sieve to count distinct prime factors
limit = 10**6
distinct_prime_count = [0] * (limit + 1)

for i in range(2, limit + 1):
    if distinct_prime_count[i] == 0:  # i is prime
        for j in range(i, limit + 1, i):
            distinct_prime_count[j] += 1

valid_numbers = []
for i in range(1, limit + 1):
    if distinct_prime_count[i] == 2:
        valid_numbers.append(i)

def solve(A):
    max_base = int(A ** 0.5)
    # Binary search for the largest number in valid_numbers that is ≤ max_base
    left, right = 0, len(valid_numbers) - 1
    result = -1
    while left <= right:
        mid = (left + right) // 2
        if valid_numbers[mid] <= max_base:
            result = mid
            left = mid + 1
        else:
            right = mid - 1
    
    if result != -1:
        return valid_numbers[result] ** 2
    else:
        return None  # Should not happen given the constraints

Q = int(input())
for _ in range(Q):
    A = int(input())
    print(solve(A))