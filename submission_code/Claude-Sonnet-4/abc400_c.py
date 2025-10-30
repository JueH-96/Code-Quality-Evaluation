import math

N = int(input())

count = 0
a = 1
while (1 << a) <= N:  # while 2^a <= N
    max_b_squared = N // (1 << a)
    max_b = int(math.sqrt(max_b_squared))
    
    # Count odd values of b from 1 to max_b
    count += (max_b + 1) // 2
    
    a += 1

print(count)