class Solution:
    def countPairs(self, nums: List[int]) -> int:
        def can_swap(s, t):
            if len(s) != len(t):
                return False
            x = int(s)
            y = int(t)
            s_list = list(s)
            for i in range(len(s_list)):
                for j in range(i + 1, len(s_list)):
                    s_list[i], s_list[j] = s_list[j], s_list[i]
                    swapped_s = ''.join(s_list)
                    if int(swapped_s) == y:
                        return True
                    s_list[i], s_list[j] = s_list[j], s_list[i]
            t_list = list(t)
            for i in range(len(t_list)):
                for j in range(i + 1, len(t_list)):
                    t_list[i], t_list[j] = t_list[j], t_list[i]
                    swapped_t = ''.join(t_list)
                    if int(swapped_t) == x:
                        return True
                    t_list[i], t_list[j] = t_list[j], s_list[j]
            return False
        
        count = 0
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                x = str(nums[i])
                y = str(nums[j])
                if can_swap(x, y):
                    count += 1
        return count