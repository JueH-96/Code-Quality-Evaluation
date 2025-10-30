# YOUR CODE HERE
def generate_repunits(limit):
    repunits = [1]
    current = 11
    while current <= limit:
        repunits.append(current)
        current = current * 10 + 1
    return repunits

def find_nth_sum_of_three_repunits(N):
    repunits = generate_repunits(10**12)  # Generate repunits up to a large limit
    sums = set()
    for i in range(len(repunits)):
        for j in range(i, len(repunits)):
            for k in range(j, len(repunits)):
                sum_repunits = repunits[i] + repunits[j] + repunits[k]
                sums.add(sum_repunits)
                if len(sums) >= N:
                    return sorted(sums)[N-1]

N = int(input())
result = find_nth_sum_of_three_repunits(N)
print(result)