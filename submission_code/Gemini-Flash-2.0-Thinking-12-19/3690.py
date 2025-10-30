class Solution:
    def minLength(self, s: str, numOps: int) -> int:
        n = len(s)
        def is_possible(max_len, current_s, ops_limit):
            s_list = list(current_s)
            ops_used = 0
            i = 0
            while i < len(s_list):
                char = s_list[i]
                j = i
                while j < len(s_list) and s_list[j] == char:
                    j += 1
                block_len = j - i
                if block_len > max_len:
                    ops_used += 1
                    if ops_used > ops_limit:
                        return False
                    if i + max_len < len(s_list):
                        s_list[i + max_len] = '1' if s_list[i + max_len] == '0' else '0'
                    else:
                        # Should not reach here in a typical case if block_len > max_len.
                        # But to handle edge cases, flip the last char of the block.
                        s_list[j-1] = '1' if s_list[j-1] == '0' else '0'
                i = j
            return True

        low = 1
        high = n
        ans = n
        while low <= high:
            mid = (low + high) // 2
            if is_possible(mid, s, numOps):
                ans = mid
                high = mid - 1
            else:
                low = mid + 1
        return ans