class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        if not words:
            return 0
        n = len(words)
        if n == 1:
            return len(words[0])
        
        # Initialize the previous state with the first word
        prev = {}
        f0, l0 = words[0][0], words[0][-1]
        prev[(f0, l0)] = len(words[0])
        
        for i in range(1, n):
            current_word = words[i]
            curr_f = current_word[0]
            curr_l = current_word[-1]
            curr_len = len(current_word)
            current = {}
            
            for (f_prev, l_prev), len_prev in prev.items():
                # Option 1: join previous string with current word
                new_f = f_prev
                new_l = curr_l
                if l_prev == curr_f:
                    new_len = len_prev + curr_len - 1
                else:
                    new_len = len_prev + curr_len
                key = (new_f, new_l)
                if key not in current or new_len < current[key]:
                    current[key] = new_len
                
                # Option 2: join current word with previous string
                new_f2 = curr_f
                new_l2 = l_prev
                if curr_f == l_prev:
                    new_len2 = curr_len + len_prev - 1
                else:
                    new_len2 = curr_len + len_prev
                key2 = (new_f2, new_l2)
                if key2 not in current or new_len2 < current.get(key2, float('inf')):
                    current[key2] = new_len2
            
            prev = current
        
        return min(prev.values())