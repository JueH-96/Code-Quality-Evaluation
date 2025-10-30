import sys
import math

def min_difference(D):
    min_diff = float('inf')
    limit = int(math.isqrt(D)) + 1
    
    for x in range(limit):
        y_squared = D - x * x
        if y_squared < 0:
            continue
        y = int(math.isqrt(y_squared))
        
        for dy in (0, 1):
            if y_squared >= dy * dy:
                current_diff = abs(x * x + y * y - D)
                min_diff = min(min_diff, current_diff)
                
    return min_diff

if __name__ == "__main__":
    D = int(sys.stdin.read().strip())
    result = min_difference(D)
    print(result)