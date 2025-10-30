class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        marked_indices = set()
        answer = []

        for query in queries:
            index_i, k_i = query

            if index_i not in marked_indices:
                marked_indices.add(index_i)

            unmarked_elements = []
            for i in range(n):
                if i not in marked_indices:
                    unmarked_elements.append((nums[i], i))

            unmarked_elements.sort(key=lambda x: (x[0], x[1]))

            count = 0
            for val, idx in unmarked_elements:
                if count < k_i:
                    if idx not in marked_indices:
                        marked_indices.add(idx)
                        count += 1
                else:
                    break

            current_sum = 0
            for i in range(n):
                if i not in marked_indices:
                    current_sum += nums[i]
            answer.append(current_sum)

        return answer