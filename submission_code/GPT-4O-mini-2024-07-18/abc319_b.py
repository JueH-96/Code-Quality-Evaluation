def generate_string(N):
    result = []
    
    # Find divisors of N that are between 1 and 9
    divisors = [j for j in range(1, 10) if N % j == 0]
    
    for i in range(N + 1):
        smallest_j = None
        
        for j in divisors:
            if i % (N // j) == 0:
                if smallest_j is None or j < smallest_j:
                    smallest_j = j
        
        if smallest_j is not None:
            result.append(str(smallest_j))
        else:
            result.append('-')
    
    return ''.join(result)

# Read input
import sys
input = sys.stdin.read
N = int(input().strip())

# Generate and print the result
print(generate_string(N))