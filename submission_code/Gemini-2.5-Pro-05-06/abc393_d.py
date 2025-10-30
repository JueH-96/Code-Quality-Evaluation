import sys

def solve():
    N = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()

    one_indices = []
    for i in range(N):
        if S[i] == '1':
            one_indices.append(i)

    k = len(one_indices)

    # Constraints: S contains at least one 1. So k >= 1.
    # If k=1, this means there's only one '1'. It's already contiguous.
    # The logic below correctly handles this:
    # one_indices = [p0], k=1
    # p_prime_values = [p0 - 0] = [p0]
    # median_idx = (1-1)//2 = 0. median_target_offset = p_prime_values[0] = p0.
    # min_total_displacement = abs(p0 - p0) = 0.
    
    p_prime_values = []
    for i in range(k):
        # one_indices[i] is the 0-indexed position of the i-th '1' (0-indexed counter for '1's)
        p_prime_values.append(one_indices[i] - i)
    
    # p_prime_values is already sorted non-decreasingly.
    # (one_indices[i+1] - (i+1)) - (one_indices[i] - i)
    # = (one_indices[i+1] - one_indices[i]) - 1
    # Since one_indices[i+1] > one_indices[i], one_indices[i+1] - one_indices[i] >= 1.
    # So, (one_indices[i+1] - one_indices[i]) - 1 >= 0.

    # The value 's_offset' that minimizes sum(|val - s_offset|) is the median of 'val's.
    # For a sorted list of k elements p_prime_values, p_prime_values[(k-1)//2] is a valid median.
    # This corresponds to:
    # - If k is odd (e.g., k=2m+1), (k-1)//2 = m. This is the unique middle element.
    # - If k is even (e.g., k=2m), (k-1)//2 = m-1. This is the lower of the two middle elements.
    #   Any s_offset between p_prime_values[m-1] and p_prime_values[m] would yield the same minimum sum.
    median_target_offset = p_prime_values[(k - 1) // 2]

    min_total_displacement = 0
    for val_p_prime in p_prime_values:
        min_total_displacement += abs(val_p_prime - median_target_offset)
    
    print(min_total_displacement)

if __name__ == "__main__":
    solve()