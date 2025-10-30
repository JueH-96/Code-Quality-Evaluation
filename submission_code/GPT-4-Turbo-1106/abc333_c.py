def find_repunit_sum(n):
    # Initialize the list of repunit sums
    repunit_sums = []
    
    # We start with the smallest repunit, which is 1
    for a in range(1, n * 3 + 1):
        for b in range(a, n * 3 + 1):
            for c in range(b, n * 3 + 1):
                # Calculate the sum of three repunits
                repunit_sum = int('1' * a) + int('1' * b) + int('1' * c)
                # Add the sum to the list if it's not already there
                if repunit_sum not in repunit_sums:
                    repunit_sums.append(repunit_sum)
    
    # Sort the list to find the N-th smallest sum
    repunit_sums.sort()
    
    # Return the N-th smallest sum
    return repunit_sums[n - 1]

# Read the input from stdin
N = int(input().strip())

# Find the N-th smallest integer that can be expressed as the sum of exactly three repunits
answer = find_repunit_sum(N)

# Write the answer to stdout
print(answer)