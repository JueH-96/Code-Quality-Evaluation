class Solution:
    def maxPossibleScore(self, start: List[int], d: int) -> int:
        def can_achieve_min_distance(mid):
            n = len(start)
            intervals = [(start[i], start[i] + d) for i in range(n)]
            intervals.sort()
            
            chosen = []
            for s, e in intervals:
                # Generate candidate values
                candidates = [s]
                for c in chosen:
                    candidates.append(c + mid)
                    candidates.append(c - mid)
                
                # Filter candidates to those within [s, e], remove duplicates, and sort
                valid_candidates = sorted(list(set([cand for cand in candidates if s <= cand <= e])))
                
                # Choose the smallest candidate that satisfies all constraints
                found = False
                for cand in valid_candidates:
                    if all(abs(cand - c) >= mid for c in chosen):
                        chosen.append(cand)
                        found = True
                        break
                
                if not found:
                    return False
            
            return True
        
        # Binary search on the answer
        left, right = 0, 2 * 10**9
        while left < right:
            mid = (left + right + 1) // 2
            if can_achieve_min_distance(mid):
                left = mid
            else:
                right = mid - 1
        
        return left