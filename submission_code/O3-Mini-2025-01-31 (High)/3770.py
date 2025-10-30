from collections import deque

class Solution:
    def generateString(self, str1: str, str2: str) -> str:
        n = len(str1)
        m = len(str2)
        L = n + m - 1
        
        # Create result array “word” and an array forced[] telling if the letter was fixed by a T-constraint.
        word = [None] * L
        forced = [False] * L
        
        # Process T constraints.
        # For each i with str1[i]=='T', for j=0..m-1 we require word[i+j]==str2[j]
        for i in range(n):
            if str1[i] == 'T':
                # The block starting at i must equal str2.
                for j in range(m):
                    pos = i + j
                    # (pos is always in range because i+m-1 <= n+m-2 = L-1)
                    if word[pos] is not None:
                        if word[pos] != str2[j]:
                            # Conflict between two T constraints.
                            return ""
                    else:
                        word[pos] = str2[j]
                        forced[pos] = True
        
        # For positions not forced, assign the default letter 'a'
        for pos in range(L):
            if word[pos] is None:
                word[pos] = 'a'
        
        # For an index i (0 <= i < n) with str1[i]=='F' we want:
        #    word[i ... i+m-1] != str2  i.e. not (word[i+j]==str2[j] for every j)
        # We will maintain an array match_count[i] = number of j (0<=j<m) with word[i+j]==str2[j].
        match_count = [0] * n  # valid only for i where str1[i]=='F'; for T windows no check required.
        for i in range(n):
            if str1[i] == 'F':
                cnt = 0
                for j in range(m):
                    pos = i + j
                    if word[pos] == str2[j]:
                        cnt += 1
                match_count[i] = cnt
                # (If there are no free (modifiable) positions in this window then they are completely forced.
                # In that case if the block equals str2 (i.e. cnt==m) we cannot break it.)
                noFree = True
                for j in range(m):
                    pos = i + j
                    if not forced[pos]:
                        noFree = False
                        break
                if noFree and cnt == m:
                    return ""
        
        # We now want to “fix” every F-window i which is violated i.e. match_count[i] == m.
        # We will use a deque for indices of F-windows that are currently “bad”.
        dq = deque()
        for i in range(n):
            if str1[i] == 'F' and match_count[i] == m:
                dq.append(i)
                
        # Define a helper function:
        # For a free (modifiable) position pos, try to "bump" it from its current letter to a new letter L_candidate.
        # We must choose L_candidate > current letter so that for every F-window j that covers pos,
        # after updating the contribution at pos, the window's match_count remains < m.
        def try_bump(pos: int) -> str:
            old_letter = word[pos]
            # Try letters from (old_letter+1) up to 'z'
            for cand in range(ord(old_letter) + 1, ord('z') + 1):
                L_candidate = chr(cand)
                valid = True
                # pos appears in every F-window j with j from max(0, pos-m+1) to min(n-1, pos)
                start = max(0, pos - m + 1)
                end = min(n - 1, pos)
                for j in range(start, end + 1):
                    if str1[j] != 'F':
                        continue
                    offset = pos - j  # in block starting at j, pos corresponds to offset
                    # For window j, determine the change.
                    old_contrib = 1 if word[pos] == str2[offset] else 0
                    new_contrib = 1 if L_candidate == str2[offset] else 0
                    # After update, window j would have: match_count[j] - old_contrib + new_contrib.
                    if match_count[j] - old_contrib + new_contrib >= m:
                        valid = False
                        break
                if valid:
                    return L_candidate
            return None
        
        # Main loop: while some F-window is violated (i.e. equals str2 completely) fix it by bumping one free cell.
        while dq:
            i = dq.popleft()
            # Sometimes, a window may have been fixed already.
            if match_count[i] < m:
                continue
            # In window i (which covers positions i ... i+m-1) all positions match str2.
            # We must choose one free (modifiable) position in that window to bump.
            candidate_found = False
            pos_to_update = None
            new_letter = None
            # To keep the overall word (of length L) lexicographically smallest,
            # we try to update a free cell as far right as possible in the block.
            for j in range(m - 1, -1, -1):
                pos = i + j
                if not forced[pos] and word[pos] == str2[j]:
                    # Try to bump this free cell.
                    cand_letter = try_bump(pos)
                    if cand_letter is not None:
                        candidate_found = True
                        pos_to_update = pos
                        new_letter = cand_letter
                        break
            if not candidate_found:
                # There is no free cell in window i that we can bump.
                return ""
            # Now update word[pos_to_update] with new_letter.
            old_letter = word[pos_to_update]
            word[pos_to_update] = new_letter
            # For every F-window j that covers pos_to_update update its match_count.
            start = max(0, pos_to_update - m + 1)
            end = min(n - 1, pos_to_update)
            for j in range(start, end + 1):
                if str1[j] != 'F':
                    continue
                offset = pos_to_update - j
                old_contrib = 1 if old_letter == str2[offset] else 0
                new_contrib = 1 if new_letter == str2[offset] else 0
                match_count[j] = match_count[j] - old_contrib + new_contrib
                if match_count[j] == m:
                    dq.append(j)
        
        return "".join(word)