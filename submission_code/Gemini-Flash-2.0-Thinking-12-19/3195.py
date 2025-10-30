class Solution:
    def minimumSteps(self, s: str) -> int:
        n = len(s)
        ones_indices = []
        for i in range(n):
            if s[i] == '1':
                ones_indices.append(i)
        
        zeros_count = 0
        for char in s:
            if char == '0':
                zeros_count += 1
        
        expected_ones_indices = []
        for i in range(len(ones_indices)):
            expected_ones_indices.append(zeros_count + i)
            
        swaps = 0
        current_ones_indices = list(ones_indices)
        
        target_s_list = ['0'] * zeros_count + ['1'] * (n - zeros_count)
        target_s = "".join(target_s_list)
        
        current_s_list = list(s)
        
        ones_positions = []
        for i in range(len(current_s_list)):
            if current_s_list[i] == '1':
                ones_positions.append(i)
                
        expected_ones_positions = []
        for i in range(len(ones_positions)):
            expected_ones_positions.append(zeros_count + i)
            
        ans = 0
        current_one_index = 0
        for i in range(n):
            if current_s_list[i] == '1':
                target_pos = zeros_count + current_one_index
                
                current_pos = i
                ans += abs(target_pos - current_pos)
                current_one_index += 1
                
        
        swaps = 0
        ones_seen = 0
        for i in range(n):
            if s[i] == '1':
                zeros_right = 0
                for j in range(i + 1, n):
                    if s[j] == '0':
                        zeros_right += 1
                swaps += zeros_right
                
        swaps = 0
        ones_count = 0
        for i in range(n):
            if s[i] == '1':
                swaps += i - ones_count
                ones_count += 1
                
        swaps = 0
        ones_pos = []
        for i in range(n):
            if s[i] == '1':
                ones_pos.append(i)
        
        target_ones_pos = []
        num_ones = len(ones_pos)
        num_zeros = n - num_ones
        for i in range(num_ones):
            target_ones_pos.append(num_zeros + i)
            
        swaps = 0
        for i in range(num_ones):
            swaps += abs(ones_pos[i] - target_ones_pos[i])
            
        swaps = 0
        ones_indices = []
        for i in range(n):
            if s[i] == '1':
                ones_indices.append(i)
        
        target_indices = []
        num_ones = len(ones_indices)
        num_zeros = n - num_ones
        for i in range(num_ones):
            target_indices.append(num_zeros + i)
            
        swaps_count = 0
        for i in range(num_ones):
            swaps_count += target_indices[i] - ones_indices[i]
            
        res = 0
        ones_count = 0
        for i in range(n):
            if s[i] == '1':
                res += i - ones_count
                ones_count += 1
        return res