import sys

def solve_query(mochi_sizes, l, r):
    n = len(mochi_sizes)
    max_kagamimochi = 0
    
    # Sort the mochi sizes in the given range
    mochi_sizes_in_range = sorted(mochi_sizes[l-1:r])
    
    # Iterate through the mochi sizes and find the maximum number of kagamimochi
    i = 0
    while i < len(mochi_sizes_in_range) - 1:
        if mochi_sizes_in_range[i] <= mochi_sizes_in_range[i+1] / 2:
            max_kagamimochi += 1
            i += 2
        else:
            i += 1
    
    return max_kagamimochi

# Read input
n = int(input())
mochi_sizes = [int(x) for x in input().split()]
q = int(input())

# Solve each query and print the result
for _ in range(q):
    l, r = [int(x) for x in input().split()]
    print(solve_query(mochi_sizes, l, r))