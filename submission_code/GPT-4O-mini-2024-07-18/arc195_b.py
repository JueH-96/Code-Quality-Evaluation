def can_make_equal_sums(N, A, B):
    # Calculate the total sum of known values in A and B
    sum_A = sum(x for x in A if x != -1)
    sum_B = sum(x for x in B if x != -1)
    
    # Count the number of -1s in A and B
    count_A_neg1 = A.count(-1)
    count_B_neg1 = B.count(-1)
    
    # Calculate the total number of pairs we need to equalize
    total_pairs = N
    
    # The total sum we need to achieve for each pair
    total_sum = sum_A + sum_B
    
    # We need to check if we can distribute the remaining values evenly
    # Each pair should sum to the same value, which means:
    # total_sum + x (where x is the sum of all replacements) should be divisible by N
    # The target sum for each pair would be (total_sum + x) / N
    # We can replace -1s with any non-negative integers, so we can adjust x accordingly
    
    # The total number of -1s we can fill
    total_neg1 = count_A_neg1 + count_B_neg1
    
    # We need to ensure that we can achieve a sum that is divisible by N
    if (total_sum % total_pairs) == 0:
        target_sum = total_sum // total_pairs
        
        # Calculate how much we need to fill in A and B to reach the target sum
        needed_A = sum(max(0, target_sum - x) for x in A if x != -1)
        needed_B = sum(max(0, target_sum - x) for x in B if x != -1)
        
        # Check if we have enough -1s to fill the needed values
        if needed_A + needed_B <= total_neg1:
            return "Yes"
    
    return "No"

import sys
input = sys.stdin.read

def main():
    data = input().splitlines()
    N = int(data[0])
    A = list(map(int, data[1].split()))
    B = list(map(int, data[2].split()))
    
    result = can_make_equal_sums(N, A, B)
    print(result)

if __name__ == "__main__":
    main()