class Solution:
    def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        # ---------------------------------------------------------------------
        # 1) Build all "layers" in O(n^2):
        #    layer[0][i] = nums[i]
        #    layer[l][i] = layer[l-1][i] XOR layer[l-1][i+1], for l >= 1
        #
        #    So layer[l] has length n-l.
        #    Then the final XOR of subarray (i..i+l) is layer[l][i].
        # ---------------------------------------------------------------------
        n = len(nums)
        layer = [ [0]*(n - l) for l in range(n) ]
        # layer[0][i] = nums[i]
        for i in range(n):
            layer[0][i] = nums[i]
        
        for l in range(1, n):
            size = n - l
            for i in range(size):
                layer[l][i] = layer[l-1][i] ^ layer[l-1][i+1]
        
        # ---------------------------------------------------------------------
        # 2) Build 1D sparse tables for each layer[l], to answer max-range queries
        #    in O(1). The array layer[l] has length = n-l.
        #    We'll store st[l][i][k] = maximum over the interval [i .. i+(1<<k)-1]
        #    of layer[l].  Then query using standard RMQ approach.
        # ---------------------------------------------------------------------
        import math
        LOG = int(math.log2(n)) + 1  # up to ~11 for n=2000
        # st[l] will be a 2D array of shape (n-l) x LOG
        st = []
        for l in range(n):
            length = n - l
            st_layer = [[0]*LOG for _ in range(length)]
            # initialize k=0
            for i in range(length):
                st_layer[i][0] = layer[l][i]
            k = 1
            step = 1
            while (step << 1) <= length:
                for i in range(length - (step << 1) + 1):
                    st_layer[i][k] = max(st_layer[i][k-1],
                                         st_layer[i + step][k-1])
                k += 1
                step <<= 1
            st.append(st_layer)
        
        # A function to get max from st[l] in the index range [L..R].
        # We assume 0 <= L <= R < len(layer[l]).
        def get_max_in_layer(l, L, R):
            length = R - L + 1
            p = length.bit_length() - 1  # floor(log2(length))
            k_size = 1 << p
            return max(st[l][L][p], st[l][R - k_size + 1][p])
        
        # ---------------------------------------------------------------------
        # 3) For each query [L, R], we check sub-subarray lengths from 1..(R-L+1).
        #    i.e. l in [0..(R-L)], and sub-subarray start i in [L..(R-l)].
        #    We want the max of layer[l][i] for i in that range.
        #    So that's a range-maximum query on st[l] from i0=L to i1=R-l.
        #    Then pick the max over all l.
        # ---------------------------------------------------------------------
        
        answers = []
        for (L, R) in queries:
            length = R - L + 1
            best = 0
            # sub-subarray length = l+1, so l from 0..(length-1)
            # sub-subarray start i in [L..R-l]
            # we do one range max query on st[l] from L..(R-l)
            for l in range(length):
                i0 = L
                i1 = R - l
                val = get_max_in_layer(l, i0, i1)
                if val > best:
                    best = val
            answers.append(best)
        
        return answers