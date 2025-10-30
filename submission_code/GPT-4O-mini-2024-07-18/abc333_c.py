def nth_repunits_sum(N):
    # Generate repunits
    repunits = [int('1' * i) for i in range(1, 20)]  # Generate repunits up to 19 digits
    
    # Set to store unique sums of three repunits
    sums = set()
    
    # Calculate sums of all combinations of three repunits (with repetition)
    for i in range(len(repunits)):
        for j in range(len(repunits)):
            for k in range(len(repunits)):
                sums.add(repunits[i] + repunits[j] + repunits[k])
    
    # Sort the sums and get the N-th smallest
    sorted_sums = sorted(sums)
    return sorted_sums[N - 1]

# Read input
import sys
input = sys.stdin.read
N = int(input().strip())

# Output the result
print(nth_repunits_sum(N))