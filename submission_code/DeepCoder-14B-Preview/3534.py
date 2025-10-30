class Solution:
    def countPairs(self, nums: List[int]) -> int:
        count = 0
        n = len(nums)
        for i in range(n):
            x = nums[i]
            for j in range(i + 1, n):
                y = nums[j]
                # Generate all possible swaps for x
                x_swaps = set()
                s = str(x)
                digits = list(s)
                for i_swap in range(len(digits)):
                    for j_swap in range(i_swap + 1, len(digits)):
                        temp = digits.copy()
                        temp[i_swap], temp[j_swap] = temp[j_swap], temp[i_swap]
                        num_str = ''.join(temp)
                        num = int(num_str)
                        x_swaps.add(num)
                x_swaps.add(x)  # include the original x
                if y in x_swaps:
                    count += 1
                    continue  # no need to check y_swaps
                # Generate all possible swaps for y
                y_swaps = set()
                s = str(y)
                digits = list(s)
                for i_swap in range(len(digits)):
                    for j_swap in range(i_swap + 1, len(digits)):
                        temp = digits.copy()
                        temp[i_swap], temp[j_swap] = temp[j_swap], temp[i_swap]
                        num_str = ''.join(temp)
                        num = int(num_str)
                        y_swaps.add(num)
                y_swaps.add(y)  # include the original y
                if x in y_swaps:
                    count += 1
        return count