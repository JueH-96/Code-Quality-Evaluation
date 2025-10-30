import math

def solve():
    n = int(input())
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]
    count = 0
    num_primes = len(primes)
    for i in range(1, 1 << num_primes):
        current_primes_subset = []
        product_of_primes = 1
        set_bits_count = 0
        for j in range(num_primes):
            if (i >> j) & 1:
                current_primes_subset.append(primes[j])
                product_of_primes *= primes[j]
                set_bits_count += 1
        
        if product_of_primes > 64: # Optimization: if product is large, floor(N^(1/product)) is likely 1 or 0. For N>=1, it's 1 unless product is too large, e.g., > log2(N). 
            root_value = 1
        else:
            low = 1
            high = n + 1
            result_root = 0
            while low < high:
                mid = (low + high) // 2
                power_val = 1
                overflow = False
                for _ in range(product_of_primes):
                    if power_val > n // mid:
                        overflow = True
                        break
                    power_val *= mid
                if overflow or power_val > n:
                    high = mid
                else:
                    result_root = mid
                    low = mid + 1
            root_value = result_root
            
        if set_bits_count % 2 == 1:
            count += root_value
        else:
            count -= root_value
            
    print(count)

if __name__ == '__main__':
    solve()