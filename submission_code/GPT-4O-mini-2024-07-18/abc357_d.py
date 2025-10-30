def remainder_of_concatenated_integer(N):
    MOD = 998244353
    str_N = str(N)
    len_N = len(str_N)
    
    # Calculate the value of V_N % MOD
    V_N_mod = 0
    for i in range(N):
        V_N_mod = (V_N_mod * (10 ** len_N) + N) % MOD
    
    return V_N_mod

# Read input
import sys
input = sys.stdin.read
N = int(input().strip())

# Get the result and print it
result = remainder_of_concatenated_integer(N)
print(result)