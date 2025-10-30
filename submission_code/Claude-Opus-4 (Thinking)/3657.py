class Solution:
    def checkValidCuts(self, n: int, rectangles: List[List[int]]) -> bool:
        def canMakeTwoCuts(intervals):
            # Get all boundary points
            points = set()
            for start, end in intervals:
                points.add(start)
                points.add(end)
            
            # Find valid cut positions
            valid_cuts = []
            for p in sorted(points):
                if 0 < p < n:
                    # Check if this is a valid cut (no interval crosses it)
                    if all(end <= p or start >= p for start, end in intervals):
                        valid_cuts.append(p)
            
            # Try all pairs of cuts
            for i in range(len(valid_cuts)):
                for j in range(i + 1, len(valid_cuts)):
                    cut1, cut2 = valid_cuts[i], valid_cuts[j]
                    
                    # Count intervals in each section
                    section1 = sum(1 for s, e in intervals if e <= cut1)
                    section2 = sum(1 for s, e in intervals if s >= cut1 and e <= cut2)
                    section3 = sum(1 for s, e in intervals if s >= cut2)
                    
                    if section1 > 0 and section2 > 0 and section3 > 0:
                        return True
            
            return False
        
        # Try horizontal cuts
        y_intervals = [(rect[1], rect[3]) for rect in rectangles]
        if canMakeTwoCuts(y_intervals):
            return True
        
        # Try vertical cuts  
        x_intervals = [(rect[0], rect[2]) for rect in rectangles]
        return canMakeTwoCuts(x_intervals)