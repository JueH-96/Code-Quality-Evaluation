def count_power_integers(N):
    power_integers = set()
    
    a = 1
    while a * a <= N:  # a must be at least 1 and a^2 must be <= N
        b = 2
        power = a ** b
        while power <= N:
            power_integers.add(power)
            b += 1
            power = a ** b
        a += 1
    
    return len(power_integers)

import sys
input = sys.stdin.read
N = int(input().strip())
result = count_power_integers(N)
print(result)