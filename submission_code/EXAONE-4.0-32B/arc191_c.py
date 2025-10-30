import random

small_primes_list = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
bases_mr = [2, 325, 9375, 28178, 450775, 9780504, 1795265022]

def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
        
    for p in small_primes_list:
        if p >= n:
            break
        if n % p == 0:
            return False
            
    d = n - 1
    s = 0
    while d % 2 == 0:
        s += 1
        d //= 2
        
    for a in bases_mr:
        if a >= n:
            a_reduced = a % n
            if a_reduced == 0:
                continue
            a_to_use = a_reduced
        else:
            a_to_use = a
            
        x = pow(a_to_use, d, n)
        if x == 1 or x == n - 1:
            continue
            
        composite_witness = True
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                composite_witness = False
                break
                
        if composite_witness:
            return False
            
    return True

def main():
    import sys
    data = sys.stdin.read().split()
    t = int(data[0])
    index = 1
    results = []
    for _ in range(t):
        N = int(data[index]); index += 1
        if N == 1:
            results.append("20250126 1")
            continue
            
        distinct_primes = []
        temp = N
        for p in small_primes_list:
            if p * p > temp:
                break
            if temp % p == 0:
                distinct_primes.append(p)
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            distinct_primes.append(temp)
            
        found_solution = False
        for k_val in range(1, 1001):
            p_val = k_val * N + 1
            if p_val > 10**18:
                break
            if p_val < 2:
                continue
            if p_val % 2 == 0 and p_val != 2:
                continue
            if is_prime(p_val):
                found_A = False
                for attempt in range(100):
                    h = random.randint(2, p_val - 1)
                    A_val = pow(h, k_val, p_val)
                    valid = True
                    for q in distinct_primes:
                        exponent = N // q
                        if pow(A_val, exponent, p_val) == 1:
                            valid = False
                            break
                    if valid:
                        results.append(f"{A_val} {p_val}")
                        found_solution = True
                        found_A = True
                        break
                if found_A:
                    break
                    
        if not found_solution:
            low_k = 1001
            high_k = (10**18 - 1) // N
            if low_k <= high_k:
                for attempt in range(100):
                    k_val = random.randint(low_k, high_k)
                    p_val = k_val * N + 1
                    if p_val > 10**18:
                        continue
                    if p_val % 2 == 0 and p_val != 2:
                        continue
                    if is_prime(p_val):
                        found_A = False
                        for att in range(100):
                            h = random.randint(2, p_val - 1)
                            A_val = pow(h, k_val, p_val)
                            valid = True
                            for q in distinct_primes:
                                exponent = N // q
                                if pow(A_val, exponent, p_val) == 1:
                                    valid = False
                                    break
                            if valid:
                                results.append(f"{A_val} {p_val}")
                                found_solution = True
                                found_A = True
                                break
                        if found_A:
                            break
            if not found_solution:
                results.append("1 1")
                
    for res in results:
        print(res)

if __name__ == "__main__":
    main()