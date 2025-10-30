class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        # Create a dictionary to store the indices of occurrences of x
        occurrence_dict = {}
        for index, num in enumerate(nums):
            if num == x:
                if x not in occurrence_dict:
                    occurrence_dict[x] = []
                occurrence_dict[x].append(index)
        
        # Process queries
        result = []
        for query in queries:
            if x in occurrence_dict and query <= len(occurrence_dict[x]):
                result.append(occurrence_dict[x][query - 1])
            else:
                result.append(-1)
        
        return result