from typing import List

class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        def is_alternating_group(subarray):
            for i in range(len(subarray) - 1):
                if subarray[i] == subarray[i + 1]:
                    return False
            return True

        def count_alternating_groups(size):
            count = 0
            for i in range(len(colors)):
                subarray = colors[i:i + size]
                if len(subarray) < size:
                    subarray += colors[:size - len(subarray)]
                if is_alternating_group(subarray):
                    count += 1
            return count

        result = []
        for query in queries:
            if query[0] == 1:
                result.append(count_alternating_groups(query[1]))
            elif query[0] == 2:
                colors[query[1]] = query[2]
        return result