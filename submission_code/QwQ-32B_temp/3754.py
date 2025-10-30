class Solution:
    def maxDistance(self, s: str, k: int) -> int:
        contrib_A, contrib_B, contrib_C, contrib_D = self.compute_contributions(s)
        max_A = self.max_prefix_sum(contrib_A, k)
        max_B = self.max_prefix_sum(contrib_B, k)
        max_C = self.max_prefix_sum(contrib_C, k)
        max_D = self.max_prefix_sum(contrib_D, k)
        return max(max_A, max_B, max_C, max_D)
    
    def compute_contributions(self, s):
        contrib_A = []
        contrib_B = []
        contrib_C = []
        contrib_D = []
        for c in s:
            # Term A: N or E → 1, else -1
            a = 1 if c in {'N', 'E'} else -1
            contrib_A.append(a)
            # Term B: E or S →1, else -1
            b = 1 if c in {'E', 'S'} else -1
            contrib_B.append(b)
            # Term C: N or W →1, else -1
            c_val = 1 if c in {'N', 'W'} else -1
            contrib_C.append(c_val)
            # Term D: W or S →1, else -1
            d = 1 if c in {'W', 'S'} else -1
            contrib_D.append(d)
        return contrib_A, contrib_B, contrib_C, contrib_D
    
    def max_prefix_sum(self, contrib, k):
        current_sum = 0
        max_sum = 0
        flips_used = 0
        for c in contrib:
            if c == 1:
                current_sum += 1
            else:
                if flips_used < k:
                    current_sum += 1  # flipped to +1
                    flips_used += 1
                else:
                    current_sum -= 1
            if current_sum > max_sum:
                max_sum = current_sum
        return max_sum