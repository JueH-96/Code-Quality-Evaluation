from bisect import bisect_left
class Solution:
    def validSequence(self, word1: str, word2: str) -> list[int]:
        n, m = len(word1), len(word2)
        # Build position lists for suffix computation
        posList = [[] for _ in range(26)]
        for i, ch in enumerate(word1):
            posList[ord(ch) - 97].append(i)
        # Build right[j]: for suffix word2[j..], the last positions matching exactly backwards
        # right[m] = n (sentinel)
        right = [ -1 ] * (m + 1)
        right[m] = n
        for j in range(m - 1, -1, -1):
            c = ord(word2[j]) - 97
            arr = posList[c]
            # find largest index in arr that is < right[j+1]
            idx = bisect_left(arr, right[j + 1]) - 1
            if idx >= 0:
                right[j] = arr[idx]
            else:
                right[j] = -1
        # Build a segment tree over word1, storing bitmask of letters in each segment
        # We can query for next match of a letter c, or next mismatch (letter != c).
        # Segment-tree size = next power of two
        size = 1
        while size < n:
            size <<= 1
        seg = [0] * (2 * size)
        # leaves
        for i in range(n):
            seg[size + i] = 1 << (ord(word1[i]) - 97)
        # build
        for i in range(size - 1, 0, -1):
            seg[i] = seg[2*i] | seg[2*i+1]
        # helpers: query first position in [ql,qr] with letter == c (match)
        # or letter != c (mismatch).
        def find_first_match(node, nl, nr, ql, qr, cmask):
            # returns index or None
            if qr < nl or nr < ql:
                return None
            if ql <= nl and nr <= qr:
                if (seg[node] & cmask) == 0:
                    return None
            if nl == nr:
                # leaf
                return nl
            mid = (nl + nr) // 2
            # go left
            left = node*2
            res = find_first_match(left, nl, mid, ql, qr, cmask)
            if res is not None:
                return res
            return find_first_match(left+1, mid+1, nr, ql, qr, cmask)
        def find_first_mismatch(node, nl, nr, ql, qr, cmask):
            # returns index or None
            if qr < nl or nr < ql:
                return None
            if ql <= nl and nr <= qr:
                # if no letter except c present, no mismatch
                if (seg[node] & ~cmask) == 0:
                    return None
            if nl == nr:
                return nl
            mid = (nl + nr) // 2
            left = node*2
            res = find_first_mismatch(left, nl, mid, ql, qr, cmask)
            if res is not None:
                return res
            return find_first_mismatch(left+1, mid+1, nr, ql, qr, cmask)
        # Now greedy build the sequence
        ans = []
        prev = -1
        mismatchUsed = False
        for j in range(m):
            c = ord(word2[j]) - 97
            cmask = 1 << c
            # compute nextMatch unbounded
            # search in [prev+1, n-1]
            l = prev + 1
            if l >= n:
                # nothing left
                return []
            # nextMatch
            nm = find_first_match(1, 0, size-1, l, n-1, cmask)
            # nextMismatch
            nm2 = find_first_mismatch(1, 0, size-1, l, n-1, cmask)
            if not mismatchUsed:
                # mismatch must leave room for perfect suffix
                valid_mm = False
                if nm2 is not None and right[j+1] != -1 and nm2 < right[j+1]:
                    valid_mm = True
                # choose match if it's better
                if valid_mm and (nm is None or nm2 < nm):
                    # pick mismatch
                    prev = nm2
                    ans.append(prev)
                    mismatchUsed = True
                elif nm is not None:
                    # pick match
                    prev = nm
                    ans.append(prev)
                elif valid_mm:
                    # fallback to mismatch
                    prev = nm2
                    ans.append(prev)
                    mismatchUsed = True
                else:
                    return []
            else:
                # mismatch already used; must perfect match remainder
                # we enforce nextMatch < right[j+1]
                if nm is None or right[j+1] == -1 or nm >= right[j+1]:
                    return []
                prev = nm
                ans.append(prev)
        return ans

# You can test with the provided examples:
if __name__ == "__main__":
    sol = Solution()
    print(sol.validSequence("vbcca", "abc"))   # [0,1,2]
    print(sol.validSequence("bacdc", "abc"))   # [1,2,4]
    print(sol.validSequence("aaaaaa", "aaabc"))# []
    print(sol.validSequence("abc", "ab"))      # [0,1]