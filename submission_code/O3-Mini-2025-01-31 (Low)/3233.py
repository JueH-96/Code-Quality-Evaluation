class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        # simulate the partition process
        def partitions_count(string: str) -> int:
            count = 0
            i = 0
            n = len(string)
            while i < n:
                distinct = {}
                j = i
                while j < n:
                    c = string[j]
                    distinct[c] = distinct.get(c, 0) + 1
                    if len(distinct) > k:
                        break
                    j += 1
                # partition is from i to j-1 if violation occurred, otherwise till n.
                count += 1
                # If violation occurred then we do not include the letter that caused violation.
                # So we set starting index for the next partition:
                if j < n and len(distinct) > k:
                    i = j
                else:
                    # j reached the end.
                    break
            return count

        best = partitions_count(s)
        n = len(s)
        letters = "abcdefghijklmnopqrstuvwxyz"
        s_list = list(s)
        # Try changing each index to any letter different from original
        for i in range(n):
            original = s_list[i]
            for ch in letters:
                if ch == original:
                    continue
                # make the change
                s_list[i] = ch
                new_str = "".join(s_list)
                cnt = partitions_count(new_str)
                if cnt > best:
                    best = cnt
                s_list[i] = original  # restore
        return best