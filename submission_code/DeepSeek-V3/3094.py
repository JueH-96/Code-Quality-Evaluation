from collections import Counter

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        count = Counter(nums)
        operations = 0
        for num, cnt in count.items():
            if cnt == 1:
                return -1
            # Try to use as many 3-operations as possible
            ops = cnt // 3
            remainder = cnt % 3
            if remainder == 0:
                operations += ops
            else:
                # If remainder is 1, we need to replace one 3-operation with two 2-operations
                # Because 3k + 1 = 3(k-1) + 4 (which is 2*2)
                # So, ops = (k-1) + 2
                if remainder == 1:
                    if ops >= 1:
                        operations += (ops - 1) + 2
                    else:
                        return -1
                else:
                    # remainder is 2
                    operations += ops + 1
        return operations