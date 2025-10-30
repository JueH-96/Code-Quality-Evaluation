# NUMBER OF TEST CASES
t = int(input())

# PROCESS EACH TEST CASE
for _ in range(t):
    # LENGTH OF THE ARRAY
    n = int(input())
    # ELEMENTS OF THE ARRAY
    arr = list(map(int, input().split()))
    
    # INITIATE VARIABLES TO STORE THE MAX SUMS FOR ODD AND EVEN POSITIONS
    max_sum_odd = arr[0] if arr[0] % 2 != 0 else 0
    max_sum_even = arr[0] if arr[0] % 2 == 0 else 0
    
    # CURRENT SUMS FOR ODD AND EVEN POSITIONS
    curr_sum_odd = arr[0] if arr[0] % 2 != 0 else 0
    curr_sum_even = arr[0] if arr[0] % 2 == 0 else 0
    
    # ITERATE THROUGH THE ARRAY
    for i in range(1, n):
        if arr[i] % 2 != 0:
            # UPDATE THE CURRENT SUM FOR ODD POSITION
            curr_sum_odd = max(curr_sum_odd + arr[i], arr[i])
            # UPDATE THE MAX SUM FOR ODD POSITION
            max_sum_odd = max(max_sum_odd, curr_sum_odd)
        else:
            # UPDATE THE CURRENT SUM FOR EVEN POSITION
            curr_sum_even = max(curr_sum_even + arr[i], arr[i])
            # UPDATE THE MAX SUM FOR EVEN POSITION
            max_sum_even = max(max_sum_even, curr_sum_even)
    
    # PRINT THE MAX SUM OF THE SUBARRAY
    print(max(max_sum_odd, max_sum_even))