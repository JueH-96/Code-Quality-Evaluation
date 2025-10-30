import math

def count_good_integers(N):
    count = 0
    max_b = int(math.isqrt(N))
    
    for b in range(1, max_b + 1):
        b_squared = b * b
        if b_squared > N:
            break
        max_a = int(math.log2(N / b_squared))
        count += max_a + 1
    
    return count

# Reading input
N = int(input().strip())

# Calculating and printing the result
result = count_good_integers(N)
print(result)