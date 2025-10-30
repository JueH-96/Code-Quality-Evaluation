def min_operations(nums1, nums2):
    n = len(nums1)
    max1 = max(nums1)
    max2 = max(nums2)
    
    # Case 1: Both maxima are already at the last index
    if max1 == nums1[-1] and max2 == nums2[-1]:
        return 0
    
    # Case 2: Check if there exists an index i (i != n-1) where nums1[i] == max1 and nums2[i] == max2
    found = False
    for i in range(n-1):
        if nums1[i] == max1 and nums2[i] == max2:
            found = True
            break
    if found:
        return 1
    
    # Case 3: Check if max1 is the last element of nums2 and max2 is the last element of nums1
    if max1 == nums2[-1] and max2 == nums1[-1]:
        return 1
    
    # Case 4: Check if both maxima are present elsewhere in their respective arrays
    has_max1_elsewhere = False
    for i in range(n-1):
        if nums1[i] == max1:
            has_max1_elsewhere = True
            break
    
    has_max2_elsewhere = False
    for j in range(n-1):
        if nums2[j] == max2:
            has_max2_elsewhere = True
            break
    
    if has_max1_elsewhere and has_max2_elsewhere:
        return 2
    
    # If none of the above cases apply, return -1
    return -1