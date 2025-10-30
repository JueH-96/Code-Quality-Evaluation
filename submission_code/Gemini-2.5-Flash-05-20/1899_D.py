import collections
import sys

def solve():
    n = int(sys.stdin.readline())
    a = list(map(int, sys.stdin.readline().split()))

    # Use collections.Counter to count frequencies of each a_i value
    counts = collections.Counter(a)

    total_pairs = 0

    # Case 1: a_i = a_j
    # If a value 'val' appears 'count' times, then 'count * (count - 1) / 2'
    # pairs can be formed from these identical values.
    for val, count in counts.items():
        if count >= 2:
            total_pairs += count * (count - 1) // 2

    # Case 2: {a_i, a_j} = {1, 2}
    # This specifically means one element is 1 and the other is 2.
    # The number of such pairs is (count of 1s) * (count of 2s).
    # .get(key, 0) is used to safely get the count, returning 0 if the key is not present.
    count_1 = counts.get(1, 0)
    count_2 = counts.get(2, 0)
    
    total_pairs += count_1 * count_2

    sys.stdout.write(str(total_pairs) + '
')

# Read the number of test cases
num_test_cases = int(sys.stdin.readline())
# Process each test case
for _ in range(num_test_cases):
    solve()