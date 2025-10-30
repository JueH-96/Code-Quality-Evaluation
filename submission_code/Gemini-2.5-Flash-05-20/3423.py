import math
from typing import List

# Define a Node class to store the four required sum states for a range
class Node:
    def __init__(self):
        # s00: max sum where left endpoint (L) is NOT chosen and right endpoint (R) is NOT chosen.
        #      Initialized to 0 because an empty subsequence is always an option, summing to 0.
        self.s00 = 0 
        
        # s01: max sum where L is NOT chosen and R IS chosen.
        #      Initialized to -infinity as this state requires an element to be chosen.
        self.s01 = -math.inf 
        
        # s10: max sum where L IS chosen and R is NOT chosen.
        #      Initialized to -infinity.
        self.s10 = -math.inf 
        
        # s11: max sum where L IS chosen and R IS chosen.
        #      Initialized to -infinity.
        self.s11 = -math.inf 

class Solution:
    def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
        N = len(nums)
        MOD = 10**9 + 7
        
        # The segment tree array. Using 1-based indexing for convenience in typical segment tree
        # structure (2*idx for left child, 2*idx+1 for right child).
        # A size of 4*N is a common safe upper bound for segment tree arrays.
        self.tree = [Node() for _ in range(4 * N)]
        
        # We'll work with a mutable copy of nums since queries update its elements.
        self.arr = list(nums) 

        # Helper function to merge two Node objects (from left and right children)
        # into a new Node representing their combined range.
        def merge(left_node: Node, right_node: Node) -> Node:
            res = Node() # This new node will store the merged results
            
            # Calculate res.s00 (L not chosen, R not chosen)
            # This state is achieved if neither nums[M] nor nums[M+1] (the adjacent elements at the merge point) are chosen.
            # Or if one of them is chosen, the adjacent one must not be.
            res.s00 = max(left_node.s00 + right_node.s00,  # M not chosen, M+1 not chosen
                          left_node.s00 + right_node.s10,  # M not chosen, M+1 chosen
                          left_node.s01 + right_node.s00)  # M chosen, M+1 not chosen
            
            # Calculate res.s01 (L not chosen, R chosen)
            # Since R is chosen, M+1 (if exists) must not be chosen.
            res.s01 = max(left_node.s00 + right_node.s01,  # M not chosen, M+1 not chosen
                          left_node.s01 + right_node.s01)  # M chosen, M+1 not chosen
            
            # Calculate res.s10 (L chosen, R not chosen)
            # Similar logic to s00.
            res.s10 = max(left_node.s10 + right_node.s00,  # M not chosen, M+1 not chosen
                          left_node.s10 + right_node.s10,  # M not chosen, M+1 chosen
                          left_node.s11 + right_node.s00)  # M chosen, M+1 not chosen
            
            # Calculate res.s11 (L chosen, R chosen)
            # Similar logic to s01.
            res.s11 = max(left_node.s10 + right_node.s01,  # M not chosen, M+1 not chosen
                          left_node.s11 + right_node.s01)  # M chosen, M+1 not chosen
            
            return res

        # Function to build the segment tree recursively
        # idx: current node's index in the self.tree array
        # L, R: range [L, R] covered by the current node
        def build(idx: int, L: int, R: int):
            if L == R: # Base case: Leaf node reached
                # For a single element nums[L]:
                self.tree[idx].s00 = 0          # If not chosen, sum is 0.
                self.tree[idx].s01 = self.arr[L] # If chosen, sum is nums[L]. It acts as rightmost.
                self.tree[idx].s10 = self.arr[L] # If chosen, sum is nums[L]. It acts as leftmost.
                self.tree[idx].s11 = self.arr[L] # If chosen, sum is nums[L]. It acts as both.
                return

            mid = (L + R) // 2
            build(2 * idx, L, mid)          # Recursively build left child
            build(2 * idx + 1, mid + 1, R)  # Recursively build right child
            
            # Merge results from children to compute current node's values
            self.tree[idx] = merge(self.tree[2 * idx], self.tree[2 * idx + 1])

        # Function to update a value in the segment tree
        # idx: current node's index
        # L, R: range covered by current node
        # pos: index of the element in self.arr to update
        # val: new value for self.arr[pos]
        def update(idx: int, L: int, R: int, pos: int, val: int):
            if L == R: # Base case: Leaf node found for the update
                self.arr[pos] = val # Update the actual array value
                # Re-initialize leaf node states with the new value
                self.tree[idx].s00 = 0
                self.tree[idx].s01 = val
                self.tree[idx].s10 = val
                self.tree[idx].s11 = val
                return

            mid = (L + R) // 2
            if pos <= mid: # The update position is in the left child's range
                update(2 * idx, L, mid, pos, val)
            else: # The update position is in the right child's range
                update(2 * idx + 1, mid + 1, R, pos, val)
            
            # After updating a child, recalculate current node's values by merging its children
            self.tree[idx] = merge(self.tree[2 * idx], self.tree[2 * idx + 1])

        # 1. Initial build of the segment tree from the initial nums array
        build(1, 0, N - 1)
        
        total_sum_of_answers = 0
        for pos_i, x_i in queries:
            # 2. For each query, update nums[pos_i] to x_i in the segment tree
            update(1, 0, N - 1, pos_i, x_i)
            
            # 3. The answer for the current query is the maximum sum over the entire array (root node).
            # It's the maximum of all four states at the root, or 0 if all are negative (empty subsequence).
            current_root_node = self.tree[1]
            current_answer = max(current_root_node.s00,
                                 current_root_node.s01,
                                 current_root_node.s10,
                                 current_root_node.s11,
                                 0) # Explicitly include 0 for the empty subsequence
            
            # 4. Add current_answer to total_sum_of_answers modulo MOD
            total_sum_of_answers = (total_sum_of_answers + current_answer) % MOD
            
        return total_sum_of_answers