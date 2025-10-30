def count_ddos_free_strings(S):
    MOD = 998244353
    n = len(S)
    
    # Precompute powers of 52 up to the maximum possible number of '?'
    max_q = S.count('?')
    power_of_52 = [1] * (max_q + 1)
    for i in range(1, max_q + 1):
        power_of_52[i] = (power_of_52[i - 1] * 52) % MOD
    
    # Count the total number of strings without restrictions
    total_strings = power_of_52[max_q]
    
    # Count the number of DDoS-type subsequences
    count_ddos = 0
    
    # We will iterate over all possible positions for the DDoS-type subsequence
    for i in range(n - 3):
        for j in range(i + 1, n - 2):
            for k in range(j + 1, n - 1):
                for l in range(k + 1, n):
                    # Check if we can form a DDoS-type string
                    if (S[i] == S[j] and S[i].isupper() and S[k].islower() and S[l].isupper()):
                        # If we have '?', we need to count the valid replacements
                        replacements = 1
                        if S[i] == '?':
                            replacements *= 26  # Can be any uppercase letter
                        if S[j] == '?':
                            replacements *= 26  # Can be any uppercase letter
                        if S[k] == '?':
                            replacements *= 26  # Can be any lowercase letter
                        if S[l] == '?':
                            replacements *= 26  # Can be any uppercase letter
                        count_ddos = (count_ddos + replacements) % MOD
    
    # The result is total strings minus those that contain DDoS-type subsequences
    result = (total_strings - count_ddos + MOD) % MOD
    return result

import sys
input = sys.stdin.read

S = input().strip()
print(count_ddos_free_strings(S))