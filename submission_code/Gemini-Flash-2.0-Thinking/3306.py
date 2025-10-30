class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        is_marked = [False] * n
        answer = []

        for index_i, k_i in queries:
            # Mark the element at index_i
            if not is_marked[index_i]:
                is_marked[index_i] = True

            # Find unmarked elements and their (value, index)
            unmarked_elements = []
            for i in range(n):
                if not is_marked[i]:
                    unmarked_elements.append((nums[i], i))

            # Sort unmarked elements by value and then index
            unmarked_elements.sort()

            # Mark k_i smallest unmarked elements
            count = 0
            for value, index in unmarked_elements:
                if count < k_i and not is_marked[index]:
                    is_marked[index] = True
                    count += 1

            # Calculate the sum of unmarked elements
            sum_unmarked = 0
            for i in range(n):
                if not is_marked[i]:
                    sum_unmarked += nums[i]

            answer.append(sum_unmarked)

        return answer