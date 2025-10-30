def solution(nums1, nums2):
    n = len(nums1)
    unique1 = set(nums1)
    unique2 = set(nums2)
    C = unique1 & unique2
    
    count1 = {}
    for num in nums1:
        if num in count1:
            count1[num] += 1
        else:
            count1[num] = 1
    
    count2 = {}
    for num in nums2:
        if num in count2:
            count2[num] += 1
        else:
            count2[num] = 1
    
    d = 0
    for x in C:
        a = count1[x]
        b = count2[x]
        if a + b > n:
            d += 1
    
    result = len(unique1) + len(unique2) - len(C) - d
    return result