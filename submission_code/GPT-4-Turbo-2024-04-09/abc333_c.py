def generate_repunits(limit):
    repunits = []
    i = 1
    while True:
        repunit = int('1' * i)
        if repunit > limit:
            break
        repunits.append(repunit)
        i += 1
    return repunits

def find_nth_sum_of_three_repunits(n):
    # Generate a large enough list of repunits
    repunits = generate_repunits(10**12)  # Arbitrarily large limit to ensure we cover enough cases
    sums_of_three_repunits = set()
    
    # Generate all possible sums of three repunits
    for i in range(len(repunits)):
        for j in range(i, len(repunits)):
            for k in range(j, len(repunits)):
                sum_repunits = repunits[i] + repunits[j] + repunits[k]
                sums_of_three_repunits.add(sum_repunits)
    
    # Sort the sums to find the N-th smallest
    sorted_sums = sorted(sums_of_three_repunits)
    return sorted_sums[n-1]

import sys
input = sys.stdin.read
N = int(input().strip())
result = find_nth_sum_of_three_repunits(N)
print(result)