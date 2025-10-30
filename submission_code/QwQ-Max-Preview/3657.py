class Solution:
    def checkValidCuts(self, n: int, rectangles: List[List[int]]) -> bool:
        # Check vertical possibility
        vertical_result = self.checkDirection(rectangles, direction='vertical')
        if vertical_result:
            return True
        
        # Check horizontal possibility
        horizontal_result = self.checkDirection(rectangles, direction='horizontal')
        return horizontal_result
    
    def checkDirection(self, rectangles, direction):
        if direction == 'vertical':
            # Check for vertical cuts
            get_start = lambda r: r[0]
            get_end = lambda r: r[2]
        else:
            # Check for horizontal cuts
            get_start = lambda r: r[1]
            get_end = lambda r: r[3]
        
        # Collect all end coordinates as candidates for x1 or y1
        candidates = set()
        for r in rectangles:
            candidates.add(get_end(r))
        candidates = sorted(candidates)
        
        for candidate in candidates:
            # Left group: end <= candidate
            left_group = [r for r in rectangles if get_end(r) <= candidate]
            if not left_group:
                continue
            # Remaining rectangles
            remaining = [r for r in rectangles if get_end(r) > candidate]
            if not remaining:
                continue
            # Middle group: start >= candidate
            middle_group = [r for r in remaining if get_start(r) >= candidate]
            if not middle_group:
                continue
            max_end_middle = max(get_end(r) for r in middle_group)
            # Right group: start >= max_end_middle
            right_group = [r for r in remaining if get_start(r) >= max_end_middle]
            if not right_group:
                continue
            min_start_right = min(get_start(r) for r in right_group)
            if max_end_middle <= min_start_right:
                return True
        return False