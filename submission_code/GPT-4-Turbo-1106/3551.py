class Solution:
    def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        # Function to compute maximum XOR subarray ending at index i
        def maxXorSubarray(nums, l, r):
            # Initialize result and current XOR
            maxXor = currXor = 0
            # Trie node structure
            trie = {}
            # Insert 0 into Trie to handle the case when subarray starts from index 0
            node = trie
            for i in range(31, -1, -1):
                bit = 0
                if not bit in node:
                    node[bit] = {}
                node = node[bit]
            # Traverse the array
            for i in range(l, r + 1):
                node = trie
                tempNode = trie
                currXor = currXor ^ nums[i]
                maxXor = max(maxXor, currXor)
                # Insert current XOR to Trie
                for j in range(31, -1, -1):
                    bit = (currXor >> j) & 1
                    if not bit in node:
                        node[bit] = {}
                    node = node[bit]
                # Calculate maximum XOR with current XOR and update result
                curr = 0
                for j in range(31, -1, -1):
                    bit = (currXor >> j) & 1
                    toggledBit = 1 - bit
                    if toggledBit in tempNode:
                        curr |= (1 << j)
                        tempNode = tempNode[toggledBit]
                    else:
                        tempNode = tempNode[bit]
                maxXor = max(maxXor, curr)
            return maxXor
        
        # Answer each query using the maxXorSubarray function
        answer = []
        for l, r in queries:
            answer.append(maxXorSubarray(nums, l, r))
        return answer