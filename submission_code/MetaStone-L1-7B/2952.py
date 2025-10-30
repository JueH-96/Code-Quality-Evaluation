def minimum1Step(nums1, nums2, x):
    sum_a = sum(nums1)
    sum_nums2 = sum(nums2)
    n = len(nums2)
    
    if sum_nums2 == 0:
        if sum_a <= x:
            return 0
        else:
            return -1
    
    nums2_sorted = sorted(nums2, reverse=True)
    prefix_nums2 = [0] * (n + 1)
    prefix_i_nums2 = [0] * (n + 1)
    
    for i in range(n):
        prefix_nums2[i + 1] = prefix_nums2[i] + nums2_sorted[i]
        prefix_i_nums2[i + 1] = prefix_i_nums2[i] + i * nums2_sorted[i]
    
    left = 0
    right = 2 * (10**18)
    
    answer = -1
    
    while left <= right:
        mid = (left + right) // 2
        max_m = min(mid, n)
        min_sum = float('inf')
        
        for m in range(0, max_m + 1):
            if m == 0:
                current_sum = sum_a + mid * sum_nums2
            else:
                part1 = (mid + 1) * prefix_nums2[m]
                part2 = prefix_i_nums2[m]
                current_sum = sum_a + mid * sum_nums2 - (part1 - part2)
            
            if current_sum < min_sum:
                min_sum = current_sum
        
        if min_sum <= x:
            answer = mid
            right = mid - 1
        else:
            left = mid + 1
    
    return answer if answer != -1 else -1