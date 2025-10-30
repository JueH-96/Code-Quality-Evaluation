# YOUR CODE HERE
import sys
input = sys.stdin.read
MOD = 998244353

def count_non_ddos_strings(s):
    n = len(s)
    q = s.count('?')
    if q == 0:
        return 1 if not contains_ddos(s) else 0
    
    # Calculate the number of ways to replace ? with any character
    total_ways = pow(52, q, MOD)
    
    # Calculate the number of ways to replace ? with a character that forms a DDoS string
    ddos_ways = 0
    if q >= 3:
        ddos_ways = pow(26, q - 2, MOD) * 26
    
    return (total_ways - ddos_ways) % MOD

def contains_ddos(s):
    n = len(s)
    for i in range(n - 3):
        if s[i] == s[i + 1] and s[i + 2].islower() and s[i + 3].isupper():
            return True
    return False

# Read input
s = input().strip()

# Output the result
print(count_non_ddos_strings(s))