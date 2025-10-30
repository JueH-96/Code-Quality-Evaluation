class Solution:
    def maximumCoins(self, coins: List[List[int]], k: int) -> int:
        from sortedcontainers import SortedDict
        
        # Create a map to hold the changes in coin count at each position
        coin_changes = SortedDict()
        
        # Populate the map with the start and end effects of each coin range
        for l, r, c in coins:
            if l in coin_changes:
                coin_changes[l] += c
            else:
                coin_changes[l] = c
            
            if r + 1 in coin_changes:
                coin_changes[r + 1] -= c
            else:
                coin_changes[r + 1] = -c
        
        # Now, we need to find the maximum sum of any k consecutive positions
        # We will iterate through the sorted keys of coin_changes and maintain a sliding window sum
        current_sum = 0
        max_sum = 0
        window = []
        
        # We need to keep track of the last position we processed to handle gaps correctly
        last_position = None
        
        for position in coin_changes.keys():
            # Before processing this position, handle any gap since the last position
            if last_position is not None and position > last_position + 1:
                # Fill the gap with the current_sum
                for _ in range(last_position + 1, position):
                    if len(window) == k:
                        current_sum -= window.pop(0)
                    window.append(current_sum)
                    max_sum = max(max_sum, current_sum)
            
            # Update the current sum with the change at this position
            current_sum += coin_changes[position]
            
            # Manage the sliding window of size k
            if len(window) == k:
                current_sum -= window.pop(0)
            window.append(current_sum)
            max_sum = max(max_sum, current_sum)
            
            # Update the last processed position
            last_position = position
        
        return max_sum