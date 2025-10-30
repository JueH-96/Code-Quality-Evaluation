def is_400_number(n):
    if n < 36:
        return False
    
    prime_factors = {}
    
    # Factorization to find prime factors and their powers
    original_n = n
    for i in range(2, int(n**0.5) + 1):
        while n % i == 0:
            if i in prime_factors:
                prime_factors[i] += 1
            else:
                prime_factors[i] = 1
            n //= i
    if n > 1:
        prime_factors[n] = 1  # n is prime itself
    
    # Check conditions for being a 400 number
    if len(prime_factors) != 2:
        return False
    
    for power in prime_factors.values():
        if power % 2 != 0:
            return False
    
    return True

def largest_400_number_not_exceeding(a):
    # Start from A and go downwards to find the largest 400 number
    for n in range(a, 35, -1):
        if is_400_number(n):
            return n
    return None  # Should never reach here due to problem constraints

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    Q = int(data[0])
    results = []
    
    for i in range(1, Q + 1):
        A = int(data[i])
        result = largest_400_number_not_exceeding(A)
        results.append(result)
    
    # Print all results
    sys.stdout.write("
".join(map(str, results)) + "
")

if __name__ == "__main__":
    main()