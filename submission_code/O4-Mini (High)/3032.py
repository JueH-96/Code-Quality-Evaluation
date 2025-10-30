class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        import array
        n = len(receiver)
        # Number of binary levels we need (max power of two ≤ k)
        L = k.bit_length()
        
        # up[j][i] will be the node reached from i after 2^j steps
        # sum_up[j][i] will be the sum of the node IDs visited in those 2^j steps
        up = []
        sum_up = []
        
        # Level 0 initialization (2^0 = 1 step)
        up0 = array.array('I', receiver)
        sum0 = array.array('Q', receiver)
        up.append(up0)
        sum_up.append(sum0)
        
        # Build binary lifting tables up to 2^(L-1)
        for j in range(1, L):
            prev_up = up[j-1]
            prev_sum = sum_up[j-1]
            cur_up = array.array('I', [0]) * n
            cur_sum = array.array('Q', [0]) * n
            for i in range(n):
                mid = prev_up[i]
                # jump from i: first go mid = up[j-1][i], then up[j-1][mid]
                cur_up[i] = prev_up[mid]
                # sum of nodes in the first half + sum in the second half
                cur_sum[i] = prev_sum[i] + prev_sum[mid]
            up.append(cur_up)
            sum_up.append(cur_sum)
        
        # Now for each starting player x, compute f(x) by decomposing k into binary
        ans = 0
        for x in range(n):
            total = x
            cur = x
            rem = k
            j = 0
            # apply jumps according to bits of k
            while rem:
                if rem & 1:
                    total += sum_up[j][cur]
                    cur = up[j][cur]
                rem >>= 1
                j += 1
            if total > ans:
                ans = total
        
        return ans