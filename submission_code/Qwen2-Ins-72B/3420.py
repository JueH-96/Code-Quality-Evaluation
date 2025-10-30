class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        # Create a dictionary to store the indices of occurrences of x
        indices = {}
        for i, num in enumerate(nums):
            if num == x:
                if x in indices:
                    indices[x].append(i)
                else:
                    indices[x] = [i]
        
        # Create a list to store the answers to the queries
        answers = []
        for query in queries:
            if x in indices and len(indices[x]) >= query:
                answers.append(indices[x][query-1])
            else:
                answers.append(-1)
        
        return answers