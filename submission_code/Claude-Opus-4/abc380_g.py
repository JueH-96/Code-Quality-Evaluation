MOD = 998244353

def modinv(a, m=MOD):
    return pow(a, m - 2, m)

def count_inversions(arr):
    count = 0
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            if arr[i] > arr[j]:
                count += 1
    return count

def count_inversions_between(arr, start, end):
    # Count inversions where both indices are in [start, end)
    count = 0
    for i in range(start, end):
        for j in range(i + 1, end):
            if arr[i] > arr[j]:
                count += 1
    return count

def count_cross_inversions(arr, start, end):
    # Count inversions where one index is in [start, end) and the other is outside
    count = 0
    n = len(arr)
    
    # Left side with shuffled region
    for i in range(start):
        for j in range(start, end):
            if arr[i] > arr[j]:
                count += 1
    
    # Shuffled region with right side
    for i in range(start, end):
        for j in range(end, n):
            if arr[i] > arr[j]:
                count += 1
    
    return count

n, k = map(int, input().split())
p = list(map(int, input().split()))

# Convert to 0-indexed
p = [x - 1 for x in p]

total_expected = 0

# For each possible starting position
for start in range(n - k + 1):
    end = start + k
    
    # Count inversions that don't involve the shuffled region
    outside_inversions = count_inversions(p) - count_inversions_between(p, start, end) - count_cross_inversions(p, start, end)
    
    # Expected inversions within the shuffled region after shuffling
    # For any pair of positions within the shuffled region, probability of inversion is 1/2
    within_expected = k * (k - 1) // 2 * modinv(2)
    
    # For cross inversions, we need to calculate expected value after shuffling
    # For each element outside and each position inside, probability depends on relative values
    cross_expected = 0
    
    # Elements to the left of shuffled region
    for i in range(start):
        # Count how many elements in the shuffled region are smaller than p[i]
        smaller_count = 0
        for j in range(start, end):
            if p[j] < p[i]:
                smaller_count += 1
        # Expected inversions from position i to the shuffled region
        cross_expected = (cross_expected + smaller_count) % MOD
    
    # Elements to the right of shuffled region
    for i in range(end, n):
        # Count how many elements in the shuffled region are larger than p[i]
        larger_count = 0
        for j in range(start, end):
            if p[j] > p[i]:
                larger_count += 1
        # Expected inversions from the shuffled region to position i
        cross_expected = (cross_expected + larger_count) % MOD
    
    # Total expected inversions for this starting position
    position_expected = (outside_inversions + within_expected + cross_expected) % MOD
    total_expected = (total_expected + position_expected) % MOD

# Divide by the number of possible starting positions
num_positions = n - k + 1
result = (total_expected * modinv(num_positions)) % MOD

print(result)