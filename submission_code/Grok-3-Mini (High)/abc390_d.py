import sys
import itertools

def all_partitions(indices):
    if not indices:
        yield ()
        return
    first_idx = indices[0]
    for size in range(1, len(indices) + 1):
        for comb in itertools.combinations(indices[1:], size - 1):
            subset = frozenset((first_idx,) + comb)
            remaining = [idx for idx in indices if idx not in subset]
            remaining.sort()
            for part in all_partitions(remaining):
                yield (subset,) + part

# Read input
data = list(map(int, sys.stdin.read().split()))
N = data[0]
A = data[1:N+1]

# Generate all partitions and compute XOR values
result_set = set()
for part in all_partitions(list(range(N))):
    xor_val = 0
    for fs in part:
        group_sum = sum(A[idx] for idx in fs)
        xor_val ^= group_sum
    result_set.add(xor_val)

# Output the number of distinct XOR values
print(len(result_set))