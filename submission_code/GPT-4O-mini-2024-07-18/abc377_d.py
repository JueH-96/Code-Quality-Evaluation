def count_valid_pairs(N, M, intervals):
    # Create a list to store the left and right bounds of the intervals
    left_bounds = []
    right_bounds = []
    
    for L, R in intervals:
        left_bounds.append(L)
        right_bounds.append(R)
    
    # Sort the intervals based on their right bounds
    right_bounds.sort()
    
    total_pairs = 0
    
    # Iterate over all possible pairs (l, r)
    for l in range(1, M + 1):
        # We need to find the maximum r such that [l, r] does not contain any [L_i, R_i]
        # This means r must be less than L_i or greater than R_i for all i
        # We can find the first interval that starts after l
        r_start = l
        
        # Count how many intervals start after l
        count_invalid = 0
        for R in right_bounds:
            if R < l:
                count_invalid += 1
            else:
                break
        
        # The number of valid r's for this l is from l to M, minus the invalid ones
        # Invalid r's are those that are in the range of [L_i, R_i] for any i
        # We need to find the first interval that starts after l
        for L in left_bounds:
            if L > l:
                break
            count_invalid += 1
        
        # Valid r's are from l to M, minus the count of invalid r's
        valid_r_count = (M - l + 1) - count_invalid
        
        if valid_r_count > 0:
            total_pairs += valid_r_count
    
    return total_pairs

import sys
input = sys.stdin.read

def main():
    data = input().strip().splitlines()
    N, M = map(int, data[0].split())
    intervals = [tuple(map(int, line.split())) for line in data[1:N + 1]]
    
    result = count_valid_pairs(N, M, intervals)
    print(result)

if __name__ == "__main__":
    main()