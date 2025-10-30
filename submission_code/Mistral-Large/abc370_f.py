import sys
from itertools import accumulate
from bisect import bisect_right

def solve():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:]))

    # Calculate the prefix sums for the cake pieces
    prefix_sum = [0] + list(accumulate(A))

    # Function to check if the minimum mass can be achieved
    def can_achieve_min_mass(min_mass):
        count = 0
        current_sum = 0
        prev_index = 0
        for i in range(1, N + 1):
            current_sum = prefix_sum[i] - prefix_sum[prev_index]
            if current_sum >= min_mass:
                count += 1
                prev_index = i
                current_sum = 0
        if count >= K:
            return True
        return False

    # Binary search for the maximum possible minimum mass
    left, right = 0, prefix_sum[N]
    while right - left > 1:
        mid = (left + right) // 2
        if can_achieve_min_mass(mid):
            left = mid
        else:
            right = mid

    min_mass = left

    # Find the number of cut lines that are never cut
    count = 0
    current_sum = 0
    prev_index = 0
    cut_lines = set()
    for i in range(1, N + 1):
        current_sum = prefix_sum[i] - prefix_sum[prev_index]
        if current_sum >= min_mass:
            count += 1
            prev_index = i
            current_sum = 0
            if count < K:
                cut_lines.add(i)

    never_cut = N - len(cut_lines)

    # Output the result
    print(min_mass, never_cut)

if __name__ == "__main__":
    solve()