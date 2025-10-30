import sys
from itertools import combinations_with_replacement

def generate_repunits(limit):
    repunits = []
    x = 1
    while x <= limit:
        repunits.append(x)
        x = x * 10 + 1
    return repunits

def find_nth_sum_of_three_repunits(N):
    limit = 10 ** 11  # A large enough number to cover the range
    repunits = generate_repunits(limit)

    sums = set()
    for a, b, c in combinations_with_replacement(repunits, 3):
        sums.add(a + b + c)

    sorted_sums = sorted(sums)
    return sorted_sums[N-1]

def main():
    input = sys.stdin.read()
    N = int(input.strip())
    result = find_nth_sum_of_three_repunits(N)
    print(result)

if __name__ == "__main__":
    main()