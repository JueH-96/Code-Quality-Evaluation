# YOUR CODE HERE
import math

def count_expressible_integers(N):
    expressible = set()
    
    for a in range(2, int(math.isqrt(N)) + 1):
        power = a * a
        while power <= N:
            expressible.add(power)
            power *= a
    
    expressible.add(1)  # 1 can be expressed as 1^b for any b >= 2
    return len(expressible)

# Read input
N = int(input().strip())

# Calculate and print the result
print(count_expressible_integers(N))