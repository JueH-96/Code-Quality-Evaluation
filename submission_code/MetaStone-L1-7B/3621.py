def minOperations(nums: List[int], k: int) -> int:
    # Check if any element is less than k
    for num in nums:
        if num < k:
            return -1
    # Sort the array
    arr = sorted(nums)
    # Extract unique elements greater than k
    unique = []
    seen = set()
    for num in arr:
        if num > k and num not in seen:
            unique.append(num)
            seen.add(num)
    # Reverse to get decreasing order
    unique = unique[::-1]
    # The number of steps is the number of unique elements greater than k
    return len(unique)