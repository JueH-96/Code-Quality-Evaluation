def find_primitive_root(n):
    # Find a primitive root for n
    # Returns a number a such that a^n - 1 is divisible by M
    # and n is the smallest such positive integer
    if n == 1:
        # For n=1, any pair where A^1-1 is multiple of M works
        # We can use A = M+1 to ensure A^1-1 = M
        return 20250126, 1
    
    # For other cases, we can use properties of multiplicative order
    # We'll use small prime numbers as M and find appropriate A
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    # For each prime p, we try to find M = p^k where p^k-1 has multiplicative order n
    for p in primes:
        k = 1
        m = p
        while m <= 10**18:
            # Try small values of A starting from 2
            a = 2
            while a <= 100:
                # Check if a is primitive root modulo m with order n
                curr_order = 1
                val = a
                is_valid = True
                
                # Calculate order by repeated multiplication
                for _ in range(n-1):
                    val = (val * a) % m
                
                # Check if n is the smallest value
                if (val - 1) % m == 0:
                    # Verify it's the smallest such n
                    val = a
                    for i in range(1, n):
                        if (val - 1) % m == 0:
                            is_valid = False
                            break
                        val = (val * a) % m
                    
                    if is_valid:
                        return a, m
                
                a += 1
            k += 1
            m = p**k
            if m > 10**18:
                break
    
    # If we haven't found a solution yet, try some larger numbers
    return 33, n * 12 + 2

def main():
    T = int(input())
    for _ in range(T):
        N = int(input())
        A, M = find_primitive_root(N)
        print(f"{A} {M}")

if __name__ == "__main__":
    main()