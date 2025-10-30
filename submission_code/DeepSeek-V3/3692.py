class Solution:
    def shortestMatchingSubstring(self, s: str, p: str) -> int:
        stars = [i for i, ch in enumerate(p) if ch == '*']
        if len(stars) != 2:
            return -1
        
        part1 = p[:stars[0]]
        part2 = p[stars[0]+1:stars[1]]
        part3 = p[stars[1]+1:]
        
        len1, len2, len3 = len(part1), len(part2), len(part3)
        n = len(s)
        min_len = float('inf')
        
        # Precompute all starting indices of part1, part2, part3 in s
        def find_all_indices(part):
            indices = []
            len_part = len(part)
            if len_part == 0:
                return indices
            for i in range(n - len_part + 1):
                if s[i:i+len_part] == part:
                    indices.append(i)
            return indices
        
        part1_indices = find_all_indices(part1)
        part2_indices = find_all_indices(part2)
        part3_indices = find_all_indices(part3)
        
        from bisect import bisect_left
        
        # Case when part2 is not empty
        if len(part2) > 0:
            for start1 in part1_indices:
                end1 = start1 + len1
                # Find the first start2 in part2_indices >= end1
                idx2 = bisect_left(part2_indices, end1)
                if idx2 < len(part2_indices):
                    start2 = part2_indices[idx2]
                    end2 = start2 + len2
                    # Find the first start3 in part3_indices >= end2
                    idx3 = bisect_left(part3_indices, end2)
                    if idx3 < len(part3_indices):
                        start3 = part3_indices[idx3]
                        current_len = start3 + len3 - start1
                        if current_len < min_len:
                            min_len = current_len
        else:
            # part2 is empty, so pattern is part1**part3
            for start1 in part1_indices:
                end1 = start1 + len1
                # Find the first start3 >= end1
                idx3 = bisect_left(part3_indices, end1)
                if idx3 < len(part3_indices):
                    start3 = part3_indices[idx3]
                    current_len = start3 + len3 - start1
                    if current_len < min_len:
                        min_len = current_len
        
        return min_len if min_len != float('inf') else -1