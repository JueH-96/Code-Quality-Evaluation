class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        sorted_elements = sorted([(val, idx) for idx, val in enumerate(nums)], key=lambda x: (x[0], x[1]))
        n = len(sorted_elements)
        index_map = {idx: pos for pos, (val, idx) in enumerate(sorted_elements)}
        total_sum = sum(nums)
        answer = []
        
        # Initialize linked list with sentinel head and tail
        head = -1
        tail = -2
        prev = {head: tail}
        next_node = {head: 0}
        prev[tail] = n - 1
        next_node[tail] = tail
        
        for i in range(n):
            prev[i] = i - 1 if i > 0 else head
            next_node[i] = i + 1 if i < n - 1 else tail
        
        for idx, k in queries:
            # Handle marking the index_i
            pos = index_map[idx]
            if pos in prev:
                # Remove from linked list
                p = prev[pos]
                nx = next_node[pos]
                next_node[p] = nx
                prev[nx] = p
                del prev[pos]
                del next_node[pos]
                total_sum -= sorted_elements[pos][0]
            
            # Handle marking k elements
            current = next_node[head]
            count = 0
            to_remove = []
            while current != tail and count < k:
                to_remove.append(current)
                current = next_node[current]
                count += 1
            
            for node in to_remove:
                if node in prev:
                    p = prev[node]
                    nx = next_node[node]
                    next_node[p] = nx
                    prev[nx] = p
                    del prev[node]
                    del next_node[node]
                    total_sum -= sorted_elements[node][0]
            
            answer.append(total_sum)
        
        return answer