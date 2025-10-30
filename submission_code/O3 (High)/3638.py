class Solution:
    def makeStringGood(self, s: str) -> int:
        """
        We only have to care about the multiset of letters, not about their order.
        
        Let cnt[i] be the amount of the i–th letter (‘a’ = 0 … ‘z’ = 25).
        
        For the target good string we have to decide two things
            • the common frequency k  (k ≥ 1)
            • for every letter whether it is present (frequency k) or absent (frequency 0)
        
        Allowed operations and their costs
            delete    : 1
            insert    : 1
            change c→c+1 : 1            (can be applied several times)
        
        Moving one character more than one step to the right is never cheaper than
        “delete + insert” (cost 2).  Therefore in an optimal solution every character
        is either
            • kept as it is,
            • deleted,
            • moved exactly one step to the next letter, or
            • created by one insertion.
        Hence characters interact only between neighbouring letters.
        
        For a fixed k we can compute the minimum cost with a small dynamic programme
        whose state only remembers whether the previous letter is kept (present) or
        deleted (absent).
        
        Definitions for a fixed k
            present : that letter has to appear k times
            absent  : that letter has to appear 0 times
        
            surplus(i, state)  – how many characters of letter i are left over
                                 (can be deleted or moved to i+1)
            shortage(i, state) – how many characters of letter i have still to be
                                 created (by insertion or by being moved from i-1)
            
            If state == “present”
                    surplus = max(0, cnt[i]-k)
                    shortage = max(0, k-cnt[i])
                    baseCost = surplus + shortage           (#deletions + #insertions)
            If state == “absent”
                    surplus = cnt[i]          (all occurrences will be deleted)
                    shortage = 0
                    baseCost = surplus        (#deletions)
        
        When the previous letter (i-1) has surplus S and the current letter (i) has
        shortage T, we may move x = min(S, T) characters from i-1 to i.
        That replaces 2 operations (delete + insert) by 1 (change), i.e. saves one
        operation.  So we simply subtract that saving from the sum of the two
        base-costs.
        
        Because the alphabet has only 26 letters, the DP for one k needs
            26 · 4 ≈ 100 primitive operations.
        k ranges from 1 … |s| ≤ 2·10⁴   →   far below one million operations overall.
        """
        n = len(s)
        cnt = [0] * 26
        for ch in s:
            cnt[ord(ch) - 97] += 1
        
        # pre-compute surplus for the “absent” choice, they never depend on k
        absentSurplus = cnt[:]              # if a letter is absent we delete everything
        absentBase    = cnt[:]              # cost == #deletions
        
        INF = 10 ** 18
        best = INF
        
        # iterate over every possible k (1 … n)
        for k in range(1, n + 1):
            # cost for letter 0
            # present
            surplus0 = max(0, cnt[0] - k)
            shortage0 = max(0, k - cnt[0])
            basePresent0 = surplus0 + shortage0
            # absent
            baseAbsent0  = absentBase[0]     # = cnt[0]
            
            dpPresent = basePresent0
            dpAbsent  = baseAbsent0
            
            prevSurplusPresent = surplus0
            prevSurplusAbsent  = absentSurplus[0]
            
            for i in range(1, 26):
                # ---- current letter as PRESENT ----
                surplusPresent = max(0, cnt[i] - k)
                shortagePresent = max(0, k - cnt[i])
                basePresent = surplusPresent + shortagePresent
                
                # shift saving from previous letter
                saveFromPrevPresent = min(prevSurplusPresent, shortagePresent)
                saveFromPrevAbsent  = min(prevSurplusAbsent,  shortagePresent)
                
                cand1 = dpPresent - saveFromPrevPresent + basePresent
                cand2 = dpAbsent  - saveFromPrevAbsent  + basePresent
                newDpPresent = cand1 if cand1 < cand2 else cand2
                
                # ---- current letter as ABSENT ----
                baseAbsent = absentBase[i]               # = cnt[i]
                # no shortage -> no shift possible
                newDpAbsent = (dpPresent if dpPresent < dpAbsent else dpAbsent) + baseAbsent
                
                # update dp values and previous-letter surpluses for next round
                dpPresent = newDpPresent
                dpAbsent  = newDpAbsent
                prevSurplusPresent = surplusPresent
                prevSurplusAbsent  = absentSurplus[i]
            
            best = min(best, dpPresent, dpAbsent)
        
        return best