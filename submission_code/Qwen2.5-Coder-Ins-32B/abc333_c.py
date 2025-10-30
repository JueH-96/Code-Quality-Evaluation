# YOUR CODE HERE
def generate_repunits(limit):
    repunits = []
    num = 1
    while num <= limit:
        repunits.append(num)
        num = num * 10 + 1
    return repunits

def find_nth_sum_of_repunits(n):
    repunits = generate_repunits(10**12)  # Generate repunits up to a large number
    sums = set()
    
    for i in range(len(repunits)):
        for j in range(i, len(repunits)):
            for k in range(j, len(repunits)):
                sums.add(repunits[i] + repunits[j] + repunits[k])
    
    sorted_sums = sorted(sums)
    return sorted_sums[n-1]

import sys
input = sys.stdin.read().strip()
n = int(input)
print(find_nth_sum_of_repunits(n))