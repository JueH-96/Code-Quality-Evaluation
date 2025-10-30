class Solution:
    def minimumBoxes(self, apple: List[int], capacity: List[int]) -> int:
        # Total number of apples we need to fit
        total_apples = sum(apple)
        
        # Sort box capacities in descending order
        capacity.sort(reverse=True)
        
        # Greedily pick largest boxes until we have enough capacity
        curr_sum = 0
        for i, cap in enumerate(capacity):
            curr_sum += cap
            if curr_sum >= total_apples:
                return i + 1
        
        # In theory we should always be able to fit (as per problem statement),
        # but just in case, return all boxes.
        return len(capacity)