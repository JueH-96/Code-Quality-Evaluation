import math

def solve():
    N = int(input())

    ans = 0
    
    # s is the exponent of 2 in X = 2^s * m^2
    # current_power_of_2 represents 2^s
    current_power_of_2 = 2  # Start with s=1, so 2^1 = 2

    while True:
        # If 2^s alone is greater than N, then 2^s * m^2 (m^2 >= 1) will also be greater.
        if current_power_of_2 > N:
            break
            
        # We need 2^s * m^2 <= N  =>  m^2 <= N / 2^s
        # max_val_for_m_squared stores N / 2^s
        max_val_for_m_squared = N // current_power_of_2
        
        # If N / 2^s is 0, then m^2 must be 0, but m must be >= 1. So no possible m.
        # This also implies that for larger s (larger current_power_of_2), N / 2^s will also be 0.
        if max_val_for_m_squared == 0:
            break 
            
        # Calculate max_m_possible = floor(sqrt(max_val_for_m_squared))
        # m must be an odd integer, 1 <= m <= max_m_possible.
        max_m_possible = 0
        if hasattr(math, 'isqrt'): # math.isqrt is available in Python 3.8+
            max_m_possible = math.isqrt(max_val_for_m_squared)
        else:
            # Robust integer square root for floor(sqrt(Y)) if math.isqrt is not available
            # k is an estimate from float sqrt
            k = int(max_val_for_m_squared**0.5)
            # Adjust k if it's too high (k*k > Y)
            while k * k > max_val_for_m_squared:
                k -= 1
            # Adjust k if it's too low ((k+1)*(k+1) <= Y)
            while (k + 1) * (k + 1) <= max_val_for_m_squared:
                k += 1
            max_m_possible = k
        
        # Number of odd integers m such that 1 <= m <= max_m_possible.
        # This is (max_m_possible + 1) // 2.
        # If max_m_possible is 0, this correctly gives 0.
        count_for_this_s = (max_m_possible + 1) // 2
        
        ans += count_for_this_s
        
        # Optimization: if current_power_of_2 > N/2, then 2*current_power_of_2 will be > N.
        # So, the next iteration's current_power_of_2 will fail the `if current_power_of_2 > N:` check.
        # We can break early.
        if current_power_of_2 > N // 2: 
            break

        # Prepare for the next value of s (effectively s_new = s_old + 1)
        current_power_of_2 *= 2

    print(ans)

# Ensure the solve function is called when the script is run
if __name__ == '__main__':
    solve()