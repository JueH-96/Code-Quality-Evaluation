def max_kagamimochi(N, A, Q, queries):
    results = []
    
    for L, R in queries:
        # Extract the relevant subarray
        subarray = A[L-1:R]  # Convert to 0-based index
        count = 0
        pairs = []
        
        # Use two pointers to find valid pairs
        i, j = 0, 1
        while j < len(subarray):
            if subarray[i] * 2 >= subarray[j]:  # A[i] can be placed on A[j]
                pairs.append((subarray[i], subarray[j]))
                count += 1
                i += 1
                j += 1
            else:
                i += 1
                if i == j:  # Ensure j is always ahead of i
                    j += 1
        
        results.append(count)
    
    return results

# Read input
import sys
input = sys.stdin.read
data = input().splitlines()

N = int(data[0])
A = list(map(int, data[1].split()))
Q = int(data[2])
queries = [tuple(map(int, line.split())) for line in data[3:3+Q]]

# Get results
results = max_kagamimochi(N, A, Q, queries)

# Print results
for result in results:
    print(result)