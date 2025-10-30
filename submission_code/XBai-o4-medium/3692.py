class Solution:
    def shortestMatchingSubstring(self, s: str, p: str) -> int:
        pre, mid, post = self.split_pattern(p)
        min_len = float('inf')
        
        def find_all_occurrences(target):
            occurrences = []
            start = 0
            while True:
                pos = s.find(target, start)
                if pos == -1:
                    break
                occurrences.append(pos)
                start = pos + 1
            return occurrences
        
        def find_str(target, start_pos):
            if not target:
                return start_pos
            return s.find(target, start_pos)
        
        if pre:
            pre_occurrences = find_all_occurrences(pre)
            for pre_start in pre_occurrences:
                mid_start = find_str(mid, pre_start + len(pre))
                if mid_start == -1:
                    continue
                post_start = find_str(post, mid_start + len(mid))
                if post_start == -1:
                    continue
                post_end = post_start + len(post) - 1
                total_length = post_end - pre_start + 1
                if total_length < min_len:
                    min_len = total_length
        else:
            mid_occurrences = find_all_occurrences(mid)
            for mid_start in mid_occurrences:
                post_start = find_str(post, mid_start + len(mid))
                if post_start == -1:
                    continue
                post_end = post_start + len(post) - 1
                total_length = post_end - mid_start + 1
                if total_length < min_len:
                    min_len = total_length
        
        return -1 if min_len == float('inf') else min_len
    
    def split_pattern(self, p: str):
        first_star = p.find('*')
        second_star = p.find('*', first_star + 1)
        pre = p[:first_star]
        mid = p[first_star+1:second_star]
        post = p[second_star+1:]
        return pre, mid, post