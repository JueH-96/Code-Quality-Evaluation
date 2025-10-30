def minOrAfterOperations(nums: List[int], k: int) -> int:
    def is_possible(m, mid):
        count = 0
        current_and = None
        for num in nums:
            if current_and is None:
                current_and = num
            else:
                current_and &= num
            if current_and <= mid:
                count += 1
                current_and = None
        return count >= m
    
    initial_or = 0
    for num in nums:
        initial_or |= num
    
    m = len(nums) - k
    low = 0
    high = initial_or
    
    while low < high:
        mid = (low + high) // 2
        if is_possible(m, mid):
            high = mid
        else:
            low = mid + 1
    return low