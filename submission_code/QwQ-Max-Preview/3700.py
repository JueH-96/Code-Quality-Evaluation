from collections import defaultdict
MOD = 10**9 + 7

class Solution:
    def subsequencesWithMiddleMode(self, nums):
        value_indices = defaultdict(list)
        for idx, num in enumerate(nums):
            value_indices[num].append(idx)
        
        result = 0
        
        for v in value_indices:
            indices = value_indices[v]
            for i in indices:
                left_part = nums[:i]
                right_part = nums[i+1:]
                
                left_v_count = left_part.count(v)
                left_non_v = [x for x in left_part if x != v]
                s = len(left_non_v)
                left_a0 = s * (s - 1) // 2 % MOD
                left_a1 = left_v_count * s % MOD
                left_a2 = left_v_count * (left_v_count - 1) // 2 % MOD
                
                right_v_count = right_part.count(v)
                right_non_v = [x for x in right_part if x != v]
                t = len(right_non_v)
                right_b0 = t * (t - 1) // 2 % MOD
                right_b1 = right_v_count * t % MOD
                right_b2 = right_v_count * (right_v_count - 1) // 2 % MOD
                
                from collections import defaultdict
                count_in_right = defaultdict(int)
                for x in right_non_v:
                    count_in_right[x] += 1
                
                sum_a = sum(count_in_right[x] for x in left_non_v)
                sum_a = sum(count_in_right.get(x, 0) for x in left_non_v)
                
                for a in range(3):
                    for b in range(3):
                        total_v = 1 + a + b
                        if total_v >= 3:
                            if a == 0:
                                left_val = left_a0
                            elif a == 1:
                                left_val = left_a1
                            else:
                                left_val = left_a2
                            if b == 0:
                                right_val = right_b0
                            elif b == 1:
                                right_val = right_b1
                            else:
                                right_val = right_b2
                            result = (result + left_val * right_val) % MOD
                        else:
                            if total_v == 1 and a == 0 and b == 0:
                                if s < 2 or t < 2:
                                    continue
                                sum_valid = 0
                                for j in range(len(left_non_v)):
                                    for k in range(j+1, len(left_non_v)):
                                        x = left_non_v[j]
                                        y = left_non_v[k]
                                        cnt_x = count_in_right.get(x, 0)
                                        cnt_y = count_in_right.get(y, 0)
                                        rem = t - cnt_x - cnt_y
                                        if rem >= 2:
                                            sum_valid += rem * (rem -1) // 2
                                sum_valid %= MOD
                                result = (result + left_a0 * right_b0 % MOD * sum_valid % MOD) % MOD
                            elif total_v == 2:
                                if a == 0 and b == 1:
                                    sum_xy = (s-1) * (s * t // 2 - sum_a) % MOD
                                    valid = right_v_count * sum_xy % MOD
                                    result = (result + left_a0 * valid) % MOD
                                elif a == 1 and b == 0:
                                    sum_x = 0
                                    for x in left_non_v:
                                        cnt = count_in_right.get(x, 0)
                                        rem = t - cnt
                                        if rem >= 2:
                                            sum_x += rem * (rem -1) // 2
                                    sum_x %= MOD
                                    valid = left_v_count * sum_x % MOD
                                    result = (result + valid * right_b0) % MOD
        return result % MOD