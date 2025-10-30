import heapq

class Solution:
    def minimizeStringValue(self, s: str) -> str:
        # Count existing letters and number of '?'
        orig_cnt = [0] * 26
        qcount = 0
        for ch in s:
            if ch == '?':
                qcount += 1
            else:
                orig_cnt[ord(ch) - ord('a')] += 1

        # Build a min‐heap of (current_count, letter_index)
        heap = [(orig_cnt[i], i) for i in range(26)]
        heapq.heapify(heap)

        # need[i] = how many '?' we assign to letter i
        need = [0] * 26

        # Greedily assign each '?' to the letter with minimal current count;
        # tie‐breaking by smaller letter index ensures lexicographic minimality
        for _ in range(qcount):
            cnt, ci = heapq.heappop(heap)
            need[ci] += 1
            # increment count and push back
            heapq.heappush(heap, (cnt + 1, ci))

        # Now fill in the '?'s in s to get the lexicographically smallest string
        # respecting the total need[] for each letter
        res = []
        used = [0] * 26  # how many we've used so far for each letter in filling

        for ch in s:
            if ch != '?':
                res.append(ch)
            else:
                # pick the smallest letter c with used[c] < need[c]
                for c in range(26):
                    if used[c] < need[c]:
                        used[c] += 1
                        res.append(chr(ord('a') + c))
                        break

        return "".join(res)