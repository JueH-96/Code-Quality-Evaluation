# YOUR CODE HERE
import math
from functools import lru_cache

@lru_cache(maxsize=None)
def log_factorial(n):
    """Calculate the logarithm of n!"""
    return sum(math.log(i) for i in range(1, n + 1))

def log_multinomial(counts):
    """Calculate the logarithm of the multinomial coefficient for the given counts"""
    total = sum(counts)
    log_result = log_factorial(total)
    for count in counts:
        if count > 0:
            log_result -= log_factorial(count)
    return log_result

def solve(n, k):
    """Find the floor((S+1)/2)-th good integer sequence"""
    counts = [k] * n
    
    # Calculate the logarithm of the total number of permutations
    log_total = log_multinomial(counts)
    
    # The logarithm of the rank we're looking for (floor((S+1)/2))
    log_target = math.log((math.exp(log_total) + 1) / 2)
    
    # Generate the sequence
    result = []
    for _ in range(n * k):
        log_cumulative = float('-inf')  # log(0)
        
        for i in range(n):
            if counts[i] > 0:
                counts[i] -= 1
                log_count = log_multinomial(counts)
                
                # Calculate cumulative sum in log space
                if log_cumulative == float('-inf'):
                    log_next = log_count
                else:
                    # log(exp(log_cumulative) + exp(log_count))
                    if log_cumulative > log_count:
                        log_next = log_cumulative + math.log1p(math.exp(log_count - log_cumulative))
                    else:
                        log_next = log_count + math.log1p(math.exp(log_cumulative - log_count))
                
                if log_next >= log_target:
                    result.append(i + 1)
                    
                    # Update log_target for the next position
                    if log_cumulative != float('-inf'):
                        if log_target > log_cumulative:
                            log_target = log_target + math.log1p(-math.exp(log_cumulative - log_target))
                        else:
                            log_target = float('-inf')  # log(0)
                    
                    break
                
                log_cumulative = log_next
                counts[i] += 1
    
    return result

# Read input
n, k = map(int, input().split())

# Solve and output
result = solve(n, k)
print(' '.join(map(str, result)))