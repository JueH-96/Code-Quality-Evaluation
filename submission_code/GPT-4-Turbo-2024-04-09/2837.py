class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        from heapq import heappop, heappush
        import sys

        if num1 == 0:
            return 0
        
        # Use a priority queue to always pick the operation that reduces num1 the most
        pq = []
        for i in range(61):
            delta = (2 ** i) + num2
            heappush(pq, (-delta, i))  # push negative because heapq is a min-heap by default

        operations = 0
        while num1 != 0:
            if not pq:
                return -1
            
            max_delta, i = heappop(pq)
            max_delta = -max_delta  # convert back to positive
            
            # Calculate the maximum number of times we can apply this operation
            count = num1 // max_delta if max_delta > 0 else (num1 + max_delta + 1) // max_delta
            
            if count > 0:
                num1 -= count * max_delta
                operations += count
            
            # If after applying the best operation, num1 is still positive and we can't reduce it further, return -1
            if num1 > 0 and (max_delta <= 0 or num1 < max_delta):
                return -1
            
            # If num1 goes negative, we need to check if we can zero it out exactly
            if num1 < 0:
                # Try to find any operation that can bring it back to zero or positive
                possible = False
                for j in range(61):
                    if (2 ** j) + num2 == -num1:
                        operations += 1
                        num1 = 0
                        possible = True
                        break
                if not possible:
                    return -1

        return operations