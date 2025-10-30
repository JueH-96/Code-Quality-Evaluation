class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        n = len(s)
        P0 = [0] * (n + 1)
        zeros_pos = []
        for i in range(n):
            if s[i] == '0':
                P0[i + 1] = P0[i] + 1
                zeros_pos.append(i)
            else:
                P0[i + 1] = P0[i]
        
        total_zeros = P0[n]
        count = 0
        
        for l in range(n):
            total_avail = total_zeros - P0[l]
            d_max = min(200, total_avail)
            
            if total_avail > 0:
                R_1 = zeros_pos[P0[l]]
            else:
                R_1 = n
            
            if R_1 > l:
                count += (R_1 - l)
            elif R_1 == n:
                count += (n - l)
            
            for d in range(1, d_max + 1):
                idx_zd = P0[l] + d - 1
                if idx_zd >= len(zeros_pos):
                    break
                R_d = zeros_pos[idx_zd]
                
                if d < d_max:
                    idx_zd_next = P0[l] + d
                    if idx_zd_next < len(zeros_pos):
                        R_next = zeros_pos[idx_zd_next]
                    else:
                        R_next = n
                else:
                    R_next = n
                
                low_bound = max(R_d, l + d * d + d - 1)
                high_bound = min(R_next - 1, n - 1)
                
                if low_bound <= high_bound:
                    count += (high_bound - low_bound + 1)
        
        return count