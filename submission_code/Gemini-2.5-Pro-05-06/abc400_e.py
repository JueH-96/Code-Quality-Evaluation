import bisect
import sys

def solve():
    # For faster I/O
    input = sys.stdin.readline
    
    Q = int(input())
    queries = [int(input()) for _ in range(Q)]

    MAX_M_VAL = 10**6

    # Sieve to find primes up to MAX_M_VAL
    is_prime = [True] * (MAX_M_VAL + 1)
    primes = []
    if MAX_M_VAL >= 0: # Check array bounds for small MAX_M_VAL
        if len(is_prime) > 0: is_prime[0] = False
        if len(is_prime) > 1: is_prime[1] = False
    
    for num in range(2, MAX_M_VAL + 1):
        if is_prime[num]:
            primes.append(num)
            # Optimization: if num*num > MAX_M_VAL, then all multiples num*k (k>=num)
            # are > MAX_M_VAL or already marked by smaller primes.
            # So, no need to iterate for multiples.
            if num > MAX_M_VAL // num: # equivalent to num * num > MAX_M_VAL, safer against overflow for fixed-size int
                continue
            for multiple in range(num * num, MAX_M_VAL + 1, num):
                is_prime[multiple] = False
    
    four_hundred_numbers_set = set()

    for i in range(len(primes)):
        p = primes[i]
        
        p_power_a = p # This is p^a, starting with a=1
        while True:
            # Condition for current p_power_a to proceed:
            # It must be possible to find a q > p such that p_power_a * q <= MAX_M_VAL.
            # Smallest q is primes[i+1].
            if i + 1 >= len(primes): # No prime q > p exists
                break
            
            # If p_power_a * primes[i+1]^1 (smallest M with this p_power_a) > MAX_M_VAL,
            # then this p_power_a is too large. Break from powers of p loop.
            if p_power_a > MAX_M_VAL // primes[i+1]:
                break

            # Inner loop for q (primes[j] where j > i)
            for j in range(i + 1, len(primes)):
                q = primes[j]
                
                # q_power_b holds q^b
                # Smallest M with this p_power_a and this q is p_power_a * q^1.
                # If p_power_a * q > MAX_M_VAL, then for this p_power_a, any q' >= q will also result in M > MAX_M_VAL.
                # So break from q loop (loop over different q's).
                if p_power_a > MAX_M_VAL // q: 
                    break 
                
                q_power_b = q # This is q^b, starting with b=1
                while True: 
                    # current_M = p_power_a * q_power_b must be <= MAX_M_VAL.
                    # Check if p_power_a * q_power_b > MAX_M_VAL
                    if p_power_a > MAX_M_VAL // q_power_b:
                        break 
                    
                    current_M = p_power_a * q_power_b
                    four_hundred_numbers_set.add(current_M * current_M)

                    # Prepare for next power of q (q_power_b * q)
                    # Check if q_power_b * q itself would overflow or exceed MAX_M_VAL
                    if q_power_b > MAX_M_VAL // q: # equivalent to q_power_b * q > MAX_M_VAL
                        break
                    # This implies next q_power_b (q^(b+1)) would be too large by itself, or
                    # p_power_a * (q_power_b * q) would be too large (current_M * q > MAX_M_VAL).
                    # The check `q_power_b > MAX_M_VAL // q` means `q^(b+1) > MAX_M_VAL`.
                    # If this is true, then `p^a * q^(b+1) > MAX_M_VAL` is also true (since p^a >= 1).
                    # So this single check is sufficient to decide if q_power_b can be multiplied by q.
                    
                    q_power_b *= q
            
            # Prepare for next power of p (p_power_a * p)
            # Check if p_power_a * p itself would exceed MAX_M_VAL
            if p_power_a > MAX_M_VAL // p: # equivalent to p_power_a * p > MAX_M_VAL
                break
            p_power_a *= p

    sorted_nums = sorted(list(four_hundred_numbers_set))
    
    ans_lines = []
    for A_val in queries:
        idx = bisect.bisect_right(sorted_nums, A_val)
        # sorted_nums[idx-1] is the largest element <= A_val
        # Problem guarantees such a number exists, so idx >= 1.
        ans_lines.append(str(sorted_nums[idx-1]))
    
    sys.stdout.write("
".join(ans_lines) + "
")

solve()