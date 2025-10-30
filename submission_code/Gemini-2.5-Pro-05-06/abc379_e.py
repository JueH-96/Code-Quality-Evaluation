import sys

def solve():
    N = int(sys.stdin.readline())
    S_str = sys.stdin.readline().strip()

    # C_k = int(S[k]) * (k+1)
    C = [0] * N
    for k in range(N):
        C[k] = int(S_str[k]) * (k + 1)

    # Term2 = sum(C_k)
    term2 = sum(C)

    # Term1 = sum_{k=0 to N-1} C_k * 10^(N-k)
    # Max C_k approx 9 * N. If N=2e5, 9N = 1.8e6 (7 digits).
    # Highest power of 10 in Term1 from C_0 * 10^N is N (C_0 is single digit).
    # Max index needed is N. Max length for coeffs array can be N + approx 10 for safety.
    # coeffs[j] stores coefficient for 10^j.
    coeffs_len = N + 15 
    coeffs = [0] * coeffs_len

    for k in range(N):
        val = C[k]
        # Unit digit of C_k contributes to power (N-k).
        # Tens digit of C_k contributes to power (N-k+1), etc.
        power_idx_for_unit_digit_of_Ck = N - k 
        
        current_relative_power = 0 # 0 for unit digit, 1 for tens, etc.
        while val > 0:
            digit = val % 10
            actual_power_idx = power_idx_for_unit_digit_of_Ck + current_relative_power
            coeffs[actual_power_idx] += digit
            val //= 10
            current_relative_power += 1
            
    # Propagate carries
    carry = 0
    for j in range(coeffs_len): # Iterate from LSB to MSB
        coeffs[j] += carry
        carry = coeffs[j] // 10
        coeffs[j] %= 10

    # Convert coeffs to Term1 (integer)
    # coeffs[0] is LSB.
    term1_str_list = []
    first_digit_found = False
    for j in range(coeffs_len - 1, -1, -1): # Iterate from MSB to LSB to build string
        if coeffs[j] != 0:
            first_digit_found = True
        if first_digit_found:
            term1_str_list.append(str(coeffs[j]))
    
    if not term1_str_list: 
        # This case implies Term1 is 0. Given constraints (digits 1-9, N>=1),
        # C_k are positive, N-k >= 1 for last C_{N-1} term.
        # So Term1 = C_0*10^N + ... + C_{N-1}*10^1 will be positive.
        # e.g. if N=1, S="1", C_0=1. Term1 = 1*10^1 = 10.
        # So term1_str_list will not be empty.
        term1 = 0 
    else:
        term1 = int("".join(term1_str_list))

    # TotalSum = (Term1 - Term2) / 9
    # Term1 - Term2 = sum C_k * (10^(N-k) - 1).
    # For N-k >= 1, (10^(N-k) - 1) is non-negative and divisible by 9.
    # Smallest N-k is 1 (for k=N-1). 10^1-1 = 9.
    # So Term1 - Term2 is non-negative and divisible by 9.
    
    total_sum = (term1 - term2) // 9
    print(total_sum)

solve()