class Solution:
    def validSequence(self, word1: str, word2: str) -> List[int]:
        n1 = len(word1)
        n2 = len(word2)
        
        result = []
        
        def check_almost_equal(s1, s2):
            if len(s1) != len(s2):
                return False
            
            diff_count = 0
            for i in range(len(s1)):
                if s1[i] != s2[i]:
                    diff_count += 1
            
            return diff_count <= 1
        
        def find_sequence(index, current_sequence, current_string):
            if len(current_string) == n2:
                if check_almost_equal(current_string, word2):
                    return current_sequence
                else:
                    return None
            
            if index >= n1:
                return None
            
            # Option 1: Include the current index
            seq1 = find_sequence(index + 1, current_sequence + [index], current_string + word1[index])
            if seq1:
                return seq1
            
            # Option 2: Exclude the current index
            seq2 = find_sequence(index + 1, current_sequence, current_string)
            if seq2:
                return seq2
            
            return None
        
        
        indices = []
        
        def solve():
            nonlocal indices
            
            def backtrack(start_index, current_list):
                nonlocal indices
                
                if len(current_list) == len(word2):
                    temp_str = ""
                    for i in current_list:
                        temp_str += word1[i]
                    
                    if check_almost_equal(temp_str, word2):
                        if not indices or current_list < indices:
                            indices = current_list[:]
                    return
                
                for i in range(start_index, len(word1)):
                    current_list.append(i)
                    backtrack(i + 1, current_list)
                    current_list.pop()
            
            backtrack(0, [])
            
        solve()
        
        return indices