# YOUR CODE HERE
import math

def count_expressible_numbers(N):
    if N == 1:
        return 1
    
    count = 0
    seen = set()
    
    for b in range(2, int(math.log2(N)) + 2):
        a = 1
        while True:
            x = a ** b
            if x > N:
                break
            if x not in seen:
                seen.add(x)
                count += 1
            a += 1
    
    return count + 1  # Adding 1 to include the number 1

N = int(input())
print(count_expressible_numbers(N))