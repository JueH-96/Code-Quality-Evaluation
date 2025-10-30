import sys

def generate_repunits(n):
    """Generate repunits up to n digits."""
    repunits = []
    for i in range(1, n + 1):
        repunit = int('1' * i)
        repunits.append(repunit)
    return repunits

def find_nth_sum(n):
    """Find the N-th smallest integer that can be expressed as the sum of exactly three repunits."""
    repunits = generate_repunits(10)  # Generate repunits up to 10 digits
    sums = set()  # Use a set to store unique sums

    for i in range(len(repunits)):
        for j in range(i, len(repunits)):
            for k in range(j, len(repunits)):
                total = repunits[i] + repunits[j] + repunits[k]
                sums.add(total)

    # Convert the set to a list and sort it
    sorted_sums = sorted(list(sums))

    # Return the N-th smallest sum
    return sorted_sums[n - 1]

# Read input from stdin
n = int(sys.stdin.readline().strip())

# Find and print the N-th smallest sum
result = find_nth_sum(n)
print(result)