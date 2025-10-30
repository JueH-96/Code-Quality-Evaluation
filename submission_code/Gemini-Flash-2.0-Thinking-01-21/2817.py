class Solution:
    def minimumCost(self, s: str) -> int:
        n = len(s)
        
        def get_cost_to_zeros(current_s):
            s_list = list(current_s)
            cost = 0
            while True:
                first_one_index = -1
                for i in range(len(s_list)):
                    if s_list[i] == '1':
                        first_one_index = i
                        break
                if first_one_index == -1:
                    break
                op_index = first_one_index
                for j in range(op_index, n):
                    s_list[j] = '1' if s_list[j] == '0' else '0'
                cost += (n - op_index)
            return cost

        def get_cost_to_ones(current_s):
            s_list = list(current_s)
            cost = 0
            while True:
                first_zero_index = -1
                for i in range(len(s_list)):
                    if s_list[i] == '0':
                        first_zero_index = i
                        break
                if first_zero_index == -1:
                    break
                op_index = first_zero_index
                for j in range(op_index + 1):
                    s_list[j] = '1' if s_list[j] == '0' else '0'
                cost += (op_index + 1)
            return cost

        cost_to_zeros = get_cost_to_zeros(s)
        cost_to_ones = get_cost_to_ones(s)
        
        return min(cost_to_zeros, cost_to_ones)