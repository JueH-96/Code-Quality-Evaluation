class Solution:
    def minAnagramLength(self, s: str) -> int:
        from collections import Counter
        count = Counter(s)
        n = len(s)
        
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                divisors = [i, n // i]
                for divisor in divisors:
                    
                    if n // divisor <= 26:
                        
                        possible_t_len = divisor
                        
                        
                        possible_t_count = Counter()
                        
                        chunk_size = n // possible_t_len
                        
                        
                        valid = True
                        for j in range(possible_t_len):
                            chunk_count = Counter()
                            for k in range(j,n,possible_t_len):
                                chunk_count[s[k]] +=1
                            
                            if j == 0:
                                possible_t_count = chunk_count
                            elif chunk_count != possible_t_count:
                                valid = False
                                break
                        if valid:
                            return possible_t_len

        return n