from collections import deque

def min_operations(nums: List[int]) -> int:
    n = len(nums)
    current_flips = 0
    queue = deque()
    operations = 0
    for i in range(n):
        if i < len(queue):
            current_flips ^= 1
        effective = nums[i] ^ current_flips
        if effective == 0:
            operations += 1
            current_flips ^= 1
            queue.append(i + 3)
            if i + 3 <= n:
                queue.append(i + 3)
        else:
            current_flips ^= 1
    # Check all elements
    for i in range(n):
        if (nums[i] ^ current_flips) != 1:
            return -1
    return operations