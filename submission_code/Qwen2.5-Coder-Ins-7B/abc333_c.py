# YOUR CODE HERE
import sys

def find_nth_repunit_sum(N):
    repunits = [1]
    current_repunit = 1
    while len(repunits) < N:
        current_repunit = current_repunit * 10 + 1
        repunits.append(current_repunit)
    
    sums = []
    for i in range(len(repunits)):
        for j in range(len(repunits)):
            for k in range(len(repunits)):
                sums.append(repunits[i] + repunits[j] + repunits[k])
    
    sums.sort()
    return sums[N-1]

# Read input
N = int(sys.stdin.read().strip())

# Find and print the N-th smallest integer that can be expressed as the sum of exactly three repunits
result = find_nth_repunit_sum(N)
print(result)