class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        """
        We can think of this problem as follows:
        
        Each element x in nums can either remain x or be incremented to x+1.
        We want the largest subset whose sorted values form a consecutive sequence.
        
        A convenient way to see this is to notice that if we want to fill a consecutive run 
        of integers v, v+1, v+2, ..., each integer v can only come from an original number 
        that is either v or v-1 (since those are the only two possibilities that can be turned into v).
        
        Therefore, we can proceed by counting how many elements are originally equal to k, for each k.
        We call this array freq, where freq[k] = number of elements equal to k in the original array.
        
        Then, we try to build the longest consecutive run by iterating over all candidate final values v 
        from 1 up to max(nums) + 1. To "use" v in the final consecutive run, we may decrement freq[v - 1] 
        if it is positive (meaning we take an original element x = v-1 and increment it to v), 
        or, if freq[v - 1] is zero, we try freq[v] (meaning we take an original element x = v and leave it at v). 
        
        If neither freq[v-1] nor freq[v] have any available count, we cannot extend the consecutive run at v 
        and so we reset and start a new run after v. We keep track of the maximum length of any such run.
        
        Example walkthrough (nums = [2,1,5,1,1]):
        
          freq[1] = 3, freq[2] = 1, freq[5] = 1 (others 0)
          
          v=1: can't use freq[0], so use freq[1] -> freq[1] becomes 2, chainLen=1
          v=2: can use freq[1] (since freq[1]>0) -> freq[1] becomes 1, chainLen=2
          v=3: can use freq[2] (since freq[2]>0) -> freq[2] becomes 0, chainLen=3
          v=4: freq[3]=0 and freq[4]=0, chain breaks, chainLen=0
          v=5: freq[4]=0 but freq[5]>0 -> freq[5] becomes 0, chainLen=1
          v=6: freq[5]=0 and freq[6]=0, chain breaks, chainLen=0
          max chain length found = 3
        """
        from collections import Counter
        
        # Count the occurrences of each value
        freq_map = Counter(nums)
        max_num = max(nums)
        
        # We'll store counts in a list for quick access
        # Make it slightly larger to handle shifts up to max_num + 1
        freq = [0] * (max_num + 2)
        for x, count in freq_map.items():
            freq[x] = count
        
        max_len = 0
        chain_len = 0
        
        # Iterate possible final values from 1 up to max_num + 1
        for v in range(1, max_num + 2):
            # If we can form v from (v-1) -> (v-1) + 1 = v
            if freq[v-1] > 0:
                freq[v-1] -= 1
                chain_len += 1
                max_len = max(max_len, chain_len)
            # else if we can form v from v -> v
            elif freq[v] > 0:
                freq[v] -= 1
                chain_len += 1
                max_len = max(max_len, chain_len)
            else:
                # we can't form this value -> break the consecutive run
                chain_len = 0
        
        return max_len