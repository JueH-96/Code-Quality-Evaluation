class Solution:
    def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        if n == 0:
            return []
        
        # Precompute max_single for all [l, r]
        max_single = [[0] * n for _ in range(n)]
        for i in range(n):
            current_max = nums[i]
            max_single[i][i] = current_max
            for j in range(i + 1, n):
                current_max = max(current_max, nums[j])
                max_single[i][j] = current_max
        
        # Precompute prefix_xor
        prefix_xor = [0] * (n + 1)
        for i in range(n):
            prefix_xor[i + 1] = prefix_xor[i] ^ nums[i]
        
        # Precompute even_xor_max for each i and j_plus_1
        even_xor_max = [[0] * (n + 1) for _ in range(n)]
        for i in range(n):
            current_max = 0
            for j_plus_1 in range(i + 1, n + 1):
                if (j_plus_1 - i) % 2 == 0:
                    current_xor = prefix_xor[i] ^ prefix_xor[j_plus_1]
                    current_max = max(current_max, current_xor)
                even_xor_max[i][j_plus_1] = current_max
            # Fill the rest with the current_max up to that point
            for j_plus_1 in range(n + 1):
                if j_plus_1 < i + 1:
                    even_xor_max[i][j_plus_1] = 0
                else:
                    if j_plus_1 > 0:
                        even_xor_max[i][j_plus_1] = max(even_xor_max[i][j_plus_1], even_xor_max[i][j_plus_1 - 1])
        
        # Precompute odd_xor_data: for each i, store sorted j and prefix_max
        odd_xor_data = []
        for i in range(n):
            j_list = []
            for j in range(i + 2, n):
                if (j - i) % 2 == 0:
                    xor_val = nums[i] ^ nums[j]
                    j_list.append((j, xor_val))
            # Sort by j
            j_list.sort()
            # Create prefix_max
            prefix_max = []
            if j_list:
                prefix_max = [0] * len(j_list)
                prefix_max[0] = j_list[0][1]
                for k in range(1, len(j_list)):
                    prefix_max[k] = max(prefix_max[k - 1], j_list[k][1])
            odd_xor_data.append((j_list, prefix_max) if j_list else None)
        
        # Process each query
        answer = []
        for l, r in queries:
            current_max = max_single[l][r] if l <= r else 0
            
            # Compute even_xor_max for [l, r]
            max_even = 0
            for i in range(l, r + 1):
                j_plus_1 = r + 1
                if j_plus_1 > n:
                    j_plus_1 = n
                current_even = even_xor_max[i][j_plus_1]
                if current_even > max_even:
                    max_even = current_even
            
            # Compute odd_xor_max for [l, r]
            max_odd = 0
            for i in range(l, r - 1 + 1):
                if i >= r - 1:
                    continue
                data = odd_xor_data[i]
                if not data:
                    continue
                j_list, prefix_max = data
                # Binary search for the largest j <= r
                low, high = 0, len(j_list) - 1
                best = -1
                while low <= high:
                    mid = (low + high) // 2
                    if j_list[mid][0] <= r:
                        best = mid
                        low = mid + 1
                    else:
                        high = mid - 1
                if best != -1:
                    current_odd = prefix_max[best]
                    if current_odd > max_odd:
                        max_odd = current_odd
            
            current_max = max(current_max, max_even, max_odd)
            answer.append(current_max)
        
        return answer