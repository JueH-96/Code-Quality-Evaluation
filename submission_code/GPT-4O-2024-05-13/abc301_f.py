# YOUR CODE HERE
MOD = 998244353

def count_valid_strings(S):
    n = len(S)
    q = S.count('?')
    
    # Precompute powers of 52 modulo MOD
    pow52 = [1] * (q + 1)
    for i in range(1, q + 1):
        pow52[i] = (pow52[i - 1] * 52) % MOD
    
    # Function to check if a substring is a DDoS-type string
    def is_ddos_type(sub):
        return sub[0] == sub[1] and sub[0].isupper() and sub[2].islower() and sub[3].isupper()
    
    # Total number of possible strings
    total_strings = pow52[q]
    
    # Count the number of invalid strings
    invalid_count = 0
    
    for i in range(n - 3):
        sub = S[i:i + 4]
        if '?' not in sub:
            if is_ddos_type(sub):
                invalid_count += pow52[q]
                invalid_count %= MOD
        else:
            # Count the number of ? in the substring
            sub_q = sub.count('?')
            if sub_q == 1:
                for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz':
                    new_sub = sub.replace('?', c)
                    if is_ddos_type(new_sub):
                        invalid_count += pow52[q - 1]
                        invalid_count %= MOD
            elif sub_q == 2:
                for c1 in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz':
                    for c2 in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz':
                        new_sub = sub.replace('?', c1, 1).replace('?', c2, 1)
                        if is_ddos_type(new_sub):
                            invalid_count += pow52[q - 2]
                            invalid_count %= MOD
            elif sub_q == 3:
                for c1 in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz':
                    for c2 in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz':
                        for c3 in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz':
                            new_sub = sub.replace('?', c1, 1).replace('?', c2, 1).replace('?', c3, 1)
                            if is_ddos_type(new_sub):
                                invalid_count += pow52[q - 3]
                                invalid_count %= MOD
            elif sub_q == 4:
                for c1 in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz':
                    for c2 in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz':
                        for c3 in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz':
                            for c4 in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz':
                                new_sub = sub.replace('?', c1, 1).replace('?', c2, 1).replace('?', c3, 1).replace('?', c4, 1)
                                if is_ddos_type(new_sub):
                                    invalid_count += pow52[q - 4]
                                    invalid_count %= MOD
    
    # Valid strings are total strings minus invalid strings
    valid_count = (total_strings - invalid_count + MOD) % MOD
    return valid_count

# Read input
import sys
input = sys.stdin.read().strip()

# Compute and print the result
print(count_valid_strings(input))