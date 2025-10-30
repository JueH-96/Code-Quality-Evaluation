class Solution:
    def maxActiveSectionsAfterTrade(self, s: str) -> int:
        n = len(s)
        base_ones = s.count('1')
        t = '1' + s + '1'
        segments = []
        cur_char = t[0]
        count = 1
        for i in range(1, len(t)):
            if t[i] == cur_char:
                count += 1
            else:
                segments.append((cur_char, count))
                cur_char = t[i]
                count = 1
        segments.append((cur_char, count))
        
        zero_segments = [cnt for ch, cnt in segments if ch == '0']
        if zero_segments:
            global_max_zero = max(zero_segments)
            count_global = zero_segments.count(global_max_zero)
            distinct_zeros = sorted(set(zero_segments), reverse=True)
            if len(distinct_zeros) >= 2:
                second_max_val = distinct_zeros[1]
            else:
                second_max_val = 0
        else:
            global_max_zero = 0
            count_global = 0
            second_max_val = 0
        
        n_seg = len(segments)
        best = base_ones
        for i in range(2, n_seg - 1, 2):
            if segments[i][0] != '1':
                continue
            a = segments[i][1]
            left_zeros = segments[i-1][1]
            right_zeros = segments[i+1][1]
            new_zero_segment = left_zeros + a + right_zeros
            
            remove_count = 0
            if left_zeros == global_max_zero:
                remove_count += 1
            if right_zeros == global_max_zero:
                remove_count += 1
            remaining_global = count_global - remove_count
            if remaining_global > 0:
                non_adjacent_max = global_max_zero
            else:
                non_adjacent_max = second_max_val
            
            max_zero_after = max(non_adjacent_max, new_zero_segment)
            gain = max_zero_after - a
            candidate = base_ones + gain
            if candidate > best:
                best = candidate
                
        return best