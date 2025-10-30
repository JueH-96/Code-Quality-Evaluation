def is_repunit(n):
    """
    Checks if a number is a repunit (a number with all digits 1).
    """
    return all(int(d) == 1 for d in str(n))

def find_nth_sum_of_three_repunits(n):
    """
    Finds the n-th smallest integer that can be expressed as the sum of exactly three repunits.
    """
    repunits = [1]
    i = 2
    while len(repunits) < n * 3:
        repunit = int('1' * i)
        repunits.append(repunit)
        i += 1

    repunits.sort()
    result = None
    for i in range(len(repunits) - 2):
        for j in range(i, len(repunits) - 1):
            for k in range(j, len(repunits)):
                candidate = repunits[i] + repunits[j] + repunits[k]
                if is_repunit(candidate):
                    if result is None or candidate < result:
                        result = candidate
                    if len(repunits) >= n * 3 and result is not None:
                        return result

    return result

# Read input from stdin
n = int(input())

# Solve the problem and print the answer to stdout
print(find_nth_sum_of_three_repunits(n))