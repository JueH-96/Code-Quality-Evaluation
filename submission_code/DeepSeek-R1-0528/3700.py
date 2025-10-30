MOD = 10**9 + 7

def nCr(n, r):
    if r == 0:
        return 1
    if r == 1:
        return n
    if r == 2:
        return n * (n - 1) // 2 if n >= 2 else 0
    return 0

class Solution:
    def subsequencesWithMiddleMode(self, nums: List[int]) -> int:
        n = len(nums)
        right_freq = {}
        for num in nums:
            right_freq[num] = right_freq.get(num, 0) + 1
        left_freq = {}
        ans = 0
        
        for j in range(n):
            x = nums[j]
            if x in right_freq:
                right_freq[x] -= 1
                if right_freq[x] == 0:
                    del right_freq[x]
            
            left_x_count = left_freq.get(x, 0)
            right_x_count = right_freq.get(x, 0)
            left_nonx_count = j - left_x_count
            right_nonx_count = (n - 1 - j) - right_x_count
            
            total_j = 0

            if left_x_count >= 1:
                ways_x = left_x_count
                same_value_pairs_right = 0
                for y, f in right_freq.items():
                    if y == x:
                        continue
                    same_value_pairs_right += f * (f - 1) // 2
                total_pairs_right = right_nonx_count * (right_nonx_count - 1) // 2 if right_nonx_count >= 2 else 0
                T_B = total_pairs_right - same_value_pairs_right
                
                ways_nonx = 0
                for y, f_left in left_freq.items():
                    if y == x:
                        continue
                    f_right = right_freq.get(y, 0)
                    subtract = f_right * (right_nonx_count - f_right)
                    ways_nonx = (ways_nonx + f_left * (T_B - subtract)) % MOD
                total_j = (total_j + ways_x * ways_nonx) % MOD

            if right_x_count >= 1:
                ways_x = right_x_count
                same_value_pairs_left = 0
                for y, f in left_freq.items():
                    if y == x:
                        continue
                    same_value_pairs_left += f * (f - 1) // 2
                total_pairs_left = left_nonx_count * (left_nonx_count - 1) // 2 if left_nonx_count >= 2 else 0
                T_A = total_pairs_left - same_value_pairs_left
                
                ways_nonx = 0
                for y, f_right in right_freq.items():
                    if y == x:
                        continue
                    f_left = left_freq.get(y, 0)
                    subtract = f_left * (left_nonx_count - f_left)
                    ways_nonx = (ways_nonx + f_right * (T_A - subtract)) % MOD
                total_j = (total_j + ways_x * ways_nonx) % MOD

            for lx in [0, 1, 2]:
                for rx in [0, 1, 2]:
                    k = lx + rx
                    if k < 2:
                        continue
                    if lx > left_x_count or rx > right_x_count:
                        continue
                    ways_x_val = nCr(left_x_count, lx) * nCr(right_x_count, rx)
                    nonx_left = nCr(left_nonx_count, 2 - lx)
                    nonx_right = nCr(right_nonx_count, 2 - rx)
                    total_j = (total_j + ways_x_val * nonx_left * nonx_right) % MOD

            ans = (ans + total_j) % MOD

            left_freq[x] = left_freq.get(x, 0) + 1

        return ans