# YOUR CODE HERE
import sys

def generate_repunits(limit):
    repunits = []
    repunit = 1
    while len(repunits) < limit:
        repunits.append(repunit)
        repunit = repunit * 10 + 1
    return repunits

def find_three_repunit_sum(n):
    repunits = generate_repunits(333)
    sums = set()

    for i in range(len(repunits)):
        for j in range(i, len(repunits)):
            for k in range(j, len(repunits)):
                sum_repunit = repunits[i] + repunits[j] + repunits[k]
                sums.add(sum_repunit)

    sorted_sums = sorted(sums)
    return sorted_sums[n-1]

def main():
    input = sys.stdin.read()
    N = int(input.strip())
    result = find_three_repunit_sum(N)
    print(result)

if __name__ == "__main__":
    main()