from itertools import accumulate

def find_max_min_mass(N, K, A):
    # Double the array to simulate the circular nature
    A = A + A
    prefix_sum = [0] + list(accumulate(A))
    
    # Binary search for the maximum minimum mass
    left, right = 0, sum(A) // K
    while left < right:
        mid = (left + right + 1) // 2
        count, temp_sum = 0, 0
        for i in range(1, len(prefix_sum)):
            if prefix_sum[i] - temp_sum >= mid:
                count += 1
                temp_sum = prefix_sum[i]
        if count >= K:
            left = mid
        else:
            right = mid - 1
    
    # Calculate the number of cut lines that are never cut
    max_min_mass = left
    count, temp_sum, never_cut = 0, 0, 0
    for i in range(1, len(prefix_sum)):
        if prefix_sum[i] - temp_sum >= max_min_mass:
            if prefix_sum[i] - temp_sum == max_min_mass:
                never_cut += 1
            count += 1
            temp_sum = prefix_sum[i]
        if count == K:
            break
    never_cut = N - never_cut
    
    return max_min_mass, never_cut

# Read input
N, K = map(int, input().split())
A = list(map(int, input().split()))

# Solve the problem
max_min_mass, never_cut = find_max_min_mass(N, K, A)

# Write output
print(max_min_mass, never_cut)