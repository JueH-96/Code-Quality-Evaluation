def could_be_equal(t_prime, s):
    len_t = len(t_prime)
    len_s = len(s)
    
    if len_t == len_s:
        # Check if they are the same or differ by exactly one character
        diff_count = sum(1 for i in range(len_t) if t_prime[i] != s[i])
        if diff_count == 0 or diff_count == 1:
            return True
    elif len_t == len_s + 1:
        # Check if t_prime can be obtained by inserting one character into s
        for i in range(len_t):
            if t_prime[:i] + t_prime[i+1:] == s:
                return True
    elif len_t + 1 == len_s:
        # Check if t_prime can be obtained by deleting one character from s
        for i in range(len_s):
            if s[:i] + s[i+1:] == t_prime:
                return True
    
    return False

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
t_prime = data[1]
strings = data[2:]

results = []
for i, s in enumerate(strings, start=1):
    if could_be_equal(t_prime, s):
        results.append(i)

print(len(results))
if results:
    print(" ".join(map(str, results)))