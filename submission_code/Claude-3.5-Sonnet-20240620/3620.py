class Solution:
    def maxDistinctElements(self, nums: List[int], k: int) -> int:
        # Count the frequency of each number
        freq = {}
        for num in nums:
            freq[num] = freq.get(num, 0) + 1
        
        # Separate single occurrences and multiple occurrences
        singles = set()
        multiples = []
        for num, count in freq.items():
            if count == 1:
                singles.add(num)
            else:
                multiples.extend([num] * (count - 1))
        
        # Sort multiples to process them in order
        multiples.sort()
        
        # Try to make multiples distinct
        for num in multiples:
            left = num - k
            right = num + k
            
            # Find the first available slot
            while left <= right:
                if left not in singles and left not in freq:
                    singles.add(left)
                    break
                left += 1
            
            # If we couldn't find a slot, we can't make this number distinct
            if left > right:
                break
        
        # Return the count of distinct elements
        return len(singles)