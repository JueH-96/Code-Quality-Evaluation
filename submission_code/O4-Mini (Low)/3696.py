class Solution:
    def countSubstrings(self, s: str) -> int:
        n = len(s)
        ans = 0
        for i, ch in enumerate(s):
            d = int(ch)
            if d == 0:
                # substrings ending in '0' cannot be divisible by 0
                continue
            # we want count of j in [0..i] so that integer(s[j..i]) % d == 0
            # we roll suffix rem backwards and detect cycle in (rem, ten_pow)
            rem = 0
            ten_pow = 1 % d
            seen = {}      # map (rem, ten_pow) -> offset where first seen
            zeros = []     # list of offsets t where rem == 0
            cycle_start = -1
            cycle_end = -1
            # walk offsets t = 1,2,... up to i+1 or until cycle
            for t in range(1, i + 2):
                # s[i-t+1 .. i], update rem
                digit = ord(s[i - t + 1]) - ord('0')
                rem = (digit * ten_pow + rem) % d
                ten_pow = (ten_pow * 10) % d
                if rem == 0:
                    zeros.append(t)
                key = (rem, ten_pow)
                if key in seen:
                    cycle_start = seen[key]
                    cycle_end = t
                    break
                seen[key] = t
            total_len = i + 1
            if cycle_start == -1:
                # no cycle: we scanned all possible substrings ending at i
                ans += len(zeros)
            else:
                # we have a cycle from offsets [cycle_start .. cycle_end-1]
                pre_len = cycle_start - 1
                cyc_len = cycle_end - cycle_start
                # count zeros in pre-cycle
                import bisect
                count_pre = bisect.bisect_right(zeros, pre_len)
                # count zeros in one cycle
                # cycle offsets are in [cycle_start, cycle_end-1]
                # find their indices in zeros
                lo = bisect.bisect_left(zeros, cycle_start)
                hi = bisect.bisect_left(zeros, cycle_end)
                count_one_cycle = hi - lo
                # remaining length after pre-cycle
                rem_len = total_len - pre_len
                full_cycles = rem_len // cyc_len
                rem_tail = rem_len % cyc_len
                # count zeros in the partial cycle
                # partial cycle covers offsets [cycle_start .. cycle_start+rem_tail-1]
                part_start = cycle_start
                part_end = cycle_start + rem_tail  # exclusive
                lo2 = bisect.bisect_left(zeros, part_start)
                hi2 = bisect.bisect_left(zeros, part_end)
                count_part = hi2 - lo2
                ans += count_pre + count_one_cycle * full_cycles + count_part
        return ans