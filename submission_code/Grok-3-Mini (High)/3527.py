from typing import List

class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        N = len(colors)
        # Initialize diff array
        diff = [0] * N
        for i in range(N):
            next_idx = (i + 1) % N
            if colors[i] != colors[next_idx]:
                diff[i] = 1
            else:
                diff[i] = 0
        
        ans = []
        for query in queries:
            if query[0] == 2:  # Change color query
                idx = query[1]
                new_color = query[2]
                # Update the color
                colors[idx] = new_color
                # Update the diff array for the affected edges
                left_edge_idx = (idx - 1) % N
                if colors[left_edge_idx] == new_color:
                    diff[left_edge_idx] = 0
                else:
                    diff[left_edge_idx] = 1
                right_edge_idx = idx
                next_tile_idx = (idx + 1) % N
                if new_color == colors[next_tile_idx]:
                    diff[right_edge_idx] = 0
                else:
                    diff[right_edge_idx] = 1
            else:  # Count query
                K = query[1]
                M = K - 1  # Number of edges
                # Sliding window to count subarrays of length M with all 1s in circular diff array
                current_sum = 0
                for i in range(M):
                    current_sum += diff[i]
                count = 0
                if current_sum == M:
                    count += 1
                for s in range(1, N):
                    remove_idx = s - 1
                    add_idx = (s + M - 1) % N
                    current_sum -= diff[remove_idx]
                    current_sum += diff[add_idx]
                    if current_sum == M:
                        count += 1
                ans.append(count)
        return ans