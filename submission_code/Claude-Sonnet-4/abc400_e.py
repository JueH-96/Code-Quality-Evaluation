import math

def sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.sqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

def generate_400_numbers():
    # Get primes up to sqrt(10^12) = 10^6
    primes = sieve(1000000)
    
    numbers_400 = set()
    max_val = 10**12
    
    # Generate all 400 numbers
    for i in range(len(primes)):
        p = primes[i]
        if p * p > max_val:
            break
            
        # Try different even powers of p
        p_power = p * p  # p^2
        while p_power <= max_val:
            for j in range(i + 1, len(primes)):
                q = primes[j]
                if p_power * q * q > max_val:
                    break
                    
                # Try different even powers of q
                q_power = q * q  # q^2
                while p_power * q_power <= max_val:
                    numbers_400.add(p_power * q_power)
                    q_power *= q * q  # Next even power
            
            p_power *= p * p  # Next even power
    
    return sorted(list(numbers_400))

def binary_search_largest_le(arr, target):
    left, right = 0, len(arr) - 1
    result = -1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] <= target:
            result = mid
            left = mid + 1
        else:
            right = mid - 1
    
    return result

# Generate all 400 numbers
numbers_400 = generate_400_numbers()

# Process queries
Q = int(input())
for _ in range(Q):
    A = int(input())
    idx = binary_search_largest_le(numbers_400, A)
    print(numbers_400[idx])