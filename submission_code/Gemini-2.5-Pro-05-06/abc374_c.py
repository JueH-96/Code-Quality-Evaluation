def solve():
    N = int(input())
    K_values = list(map(int, input().split()))

    total_sum_K = sum(K_values)
    
    # Initialize min_max_group_sum with a value that is definitely an upper bound.
    # One such value is total_sum_K itself (achieved if one group is empty).
    min_max_group_sum = total_sum_K

    # There are 2^N possible ways to assign N departments to two groups.
    # We can represent an assignment using a bitmask of length N.
    # Iterate through all possible masks from 0 to 2^N - 1.
    num_combinations = 1 << N  # This is 2^N
    
    for i in range(num_combinations):
        current_sum_A = 0
        # Calculate sum for Group A based on the current mask i
        # For each department K_values[j]:
        # If the j-th bit in the mask i is 1, it's assigned to Group A.
        for j in range(N):
            if (i >> j) & 1: # Check if j-th bit of i is set
                current_sum_A += K_values[j]
        
        # The sum for Group B is the total sum minus sum for Group A
        current_sum_B = total_sum_K - current_sum_A
        
        # We are interested in the maximum of the two group sums for this particular assignment
        current_max_for_this_combination = max(current_sum_A, current_sum_B)
        
        # Update the overall minimum if this combination yields a smaller maximum
        if current_max_for_this_combination < min_max_group_sum:
            min_max_group_sum = current_max_for_this_combination
            
    print(min_max_group_sum)

if __name__ == '__main__':
    solve()