class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        # Create a dictionary to store the indices of each occurrence of x
        indices = {}
        for i, num in enumerate(nums):
            if num == x:
                if num not in indices:
                    indices[num] = []
                indices[num].append(i)
        
        # Iterate through the queries and find the answer for each query
        answer = []
        for query in queries:
            if x not in indices or len(indices[x]) < query:
                answer.append(-1)
            else:
                answer.append(indices[x][query - 1])
        
        return answer