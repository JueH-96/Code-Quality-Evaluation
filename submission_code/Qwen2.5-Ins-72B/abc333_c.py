# YOUR CODE HERE
def generate_repunits(limit):
    repunits = []
    current = 1
    while current <= limit:
        repunits.append(current)
        current = current * 10 + 1
    return repunits

def find_nth_sum(n):
    repunits = generate_repunits(10**10)
    sums = set()
    
    for i in range(len(repunits)):
        for j in range(i, len(repunits)):
            for k in range(j, len(repunits)):
                sums.add(repunits[i] + repunits[j] + repunits[k])
    
    sorted_sums = sorted(sums)
    return sorted_sums[n-1]

N = int(input())
print(find_nth_sum(N))