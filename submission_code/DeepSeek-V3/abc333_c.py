# YOUR CODE HERE
def generate_repunits(limit):
    repunits = []
    current = 1
    while current < limit:
        repunits.append(current)
        current = current * 10 + 1
    return repunits

def find_nth_sum_of_three_repunits(N):
    # We need to find the N-th smallest sum of three repunits
    # To do this, we need to generate all possible sums of three repunits and sort them
    # Since the repunits grow exponentially, we can limit the number of repunits we consider
    # For example, up to 10^12 should be sufficient for N up to 333
    
    # Generate repunits up to a certain limit
    limit = 10**12
    repunits = generate_repunits(limit)
    
    # Generate all possible sums of three repunits
    sums = set()
    for a in repunits:
        for b in repunits:
            for c in repunits:
                s = a + b + c
                if s > limit:
                    break
                sums.add(s)
            if a + b + repunits[0] > limit:
                break
        if a + repunits[0] + repunits[0] > limit:
            break
    
    # Sort the sums and find the N-th smallest
    sorted_sums = sorted(sums)
    return sorted_sums[N-1]

# Read input
N = int(input())
# Compute the result
result = find_nth_sum_of_three_repunits(N)
# Print the result
print(result)