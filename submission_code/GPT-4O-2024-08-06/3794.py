class Solution:
    def minTime(self, skill: List[int], mana: List[int]) -> int:
        n = len(skill)
        m = len(mana)
        
        # Initialize the time matrix
        time = [[0] * m for _ in range(n)]
        
        # Calculate the time taken by each wizard for each potion
        for i in range(n):
            for j in range(m):
                time[i][j] = skill[i] * mana[j]
        
        # Initialize the completion time for each wizard
        completion_time = [0] * n
        
        # Process each potion
        for j in range(m):
            # The start time for the first wizard is the maximum of the current completion time
            # of the first wizard and the completion time of the last wizard for the previous potion
            completion_time[0] = max(completion_time[0], completion_time[n - 1] if j > 0 else 0) + time[0][j]
            
            # Calculate the completion time for each wizard for the current potion
            for i in range(1, n):
                completion_time[i] = max(completion_time[i], completion_time[i - 1]) + time[i][j]
        
        # The total time is the completion time of the last wizard for the last potion
        return completion_time[n - 1]