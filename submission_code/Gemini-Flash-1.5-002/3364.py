class Solution:
    def minimumValueSum(self, nums: List[int], andValues: List[int]) -> int:
        n = len(nums)
        m = len(andValues)

        def check(partitions):
            cur_and = nums[0]
            start = 0
            count = 0
            for i in range(n):
                cur_and &= nums[i]
                if i + 1 in partitions:
                    if cur_and != andValues[count]:
                        return False
                    count += 1
                    cur_and = nums[i+1] if i + 1 < n else 0
            if count != m:
                return False
            return True

        def find_partitions(index, current_partition, partitions):
            if index == n:
                if len(partitions) == m:
                    if check(partitions):
                        return partitions
                return None

            res = None
            # Option 1: Add current element to the last partition
            if len(current_partition) > 0:
                new_partition = current_partition + [index + 1]
                result = find_partitions(index + 1, new_partition, partitions)
                if result:
                    res = result
            # Option 2: Start a new partition
            new_partitions = partitions + [index + 1]
            result = find_partitions(index + 1, [index + 1], new_partitions)
            if result:
                if res is None or len(result) < len(res):
                    res = result
            return res

        partitions = find_partitions(0, [], [])
        if partitions is None:
            return -1
        
        total_sum = 0
        last_index = 0
        for i in range(m):
            total_sum += nums[partitions[i]-1]

        return total_sum