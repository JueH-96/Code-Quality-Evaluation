from itertools import combinations_with_replacement

def nth_sum_of_three_repunits(N):
    # Generate a list of repunits up to a reasonable size
    repunits = []
    for i in range(1, 16):  # 15 repunits should be enough for N up to 333
        repunits.append((10**i - 1) // 9)
    
    # Generate all sums of three repunits
    sums = []
    for combo in combinations_with_replacement(repunits, 3):
        sums.append(sum(combo))
    
    # Sort the sums
    sums.sort()
    
    # Return the Nth sum
    return sums[N-1]

N = int(input())
print(nth_sum_of_three_repunits(N))