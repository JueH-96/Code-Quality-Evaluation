import sys
import math

MOD = 998244353

def get_divisors(x):
    divs = set()
    for i in range(1, int(math.isqrt(x)) + 1):
        if x % i == 0:
            divs.add(i)
            divs.add(x // i)
    return list(divs)

def main():
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    max_A = 10**5  # since A_i up to 1e5
    max_mu = 10**5 + 1
    
    # Precompute mobius function
    mu = [1] * (max_mu)
    is_prime = [True] * (max_mu)
    for i in range(2, max_mu):
        if is_prime[i]:
            # For all multiples of i, multiply mu by -1
            for j in range(i, max_mu, i):
                is_prime[j] = False
                mu[j] *= -1
            # For all multiples of i^2, set mu to 0
            j = i * i
            while j < max_mu:
                mu[j] = 0
                j += i * i
    
    # Initialize cnt and mul_sum
    max_val = 10**5 + 1
    cnt = [0] * (max_val)
    mul_sum = [0] * (max_val)
    total_sum = 0
    total_count = 0
    answers = []
    
    for a in A:
        divs_a = get_divisors(a)
        sum_gcd = 0
        
        for d in divs_a:
            a_prime = a // d
            divs_a_prime = get_divisors(a_prime)
            current_term = 0
            for k in divs_a_prime:
                product = d * k
                if product >= max_val:
                    contrib = 0
                else:
                    contrib = mu[k] * mul_sum[product]
                current_term += contrib
            sum_gcd += d * current_term
        
        new_total_sum = (2 * total_sum + sum_gcd) % MOD
        new_total_count = 2 * total_count + 1
        
        delta = (total_count + 1) - cnt[a]
        cnt[a] += (total_count + 1)
        
        for divisor in divs_a:
            mul_sum[divisor] += delta
        
        total_sum = new_total_sum
        total_count = new_total_count
        
        answers.append(total_sum % MOD)
    
    for ans in answers:
        print(ans)
        
if __name__ == "__main__":
    main()