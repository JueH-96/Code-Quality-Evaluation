# YOUR CODE HERE
def generate_repunits(limit):
    repunits = []
    num = 1
    while len(repunits) < limit:
        repunits.append(num)
        num = num * 10 + 1
    return repunits

def find_nth_sum_of_three_repunits(N):
    repunits = generate_repunits(20)  # Generate first 20 repunits, which is more than enough
    sums = set()
    
    for i in range(len(repunits)):
        for j in range(len(repunits)):
            for k in range(len(repunits)):
                sums.add(repunits[i] + repunits[j] + repunits[k])
    
    sorted_sums = sorted(sums)
    return sorted_sums[N-1]

import sys
input = sys.stdin.read
N = int(input().strip())
print(find_nth_sum_of_three_repunits(N))