# YOUR CODE HERE

def find_nth_repunit_sum(n):
    repunits = [1, 11, 111]
    sums = set()

    for i in repunits:
        for j in repunits:
            for k in repunits:
                sums.add(i + j + k)

    sums = sorted(list(sums))
    return sums[n - 1]

n = int(input())
print(find_nth_repunit_sum(n))