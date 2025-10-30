class Solution:
    def possibleStringCount(self, word: str) -> int:
        # We will split the string into consecutive-character runs.
        # For a run of length k, if that run were the "mistake" run (typed too long),
        # the original could have had between 1 and k-1 occurrences of that character.
        # That gives (k-1) possible originals for that run.
        #
        # We can only have at most one such mistake in the entire string, so for each run,
        # if we pick that run as the mistake run, we get (k - 1) possibilities.
        #
        # Therefore, the total possible originals is:
        #   1 (no mistake anywhere) + sum( (k_i - 1) for each run_i )
        # because picking different runs leads to distinct original strings.
        
        if not word:
            return 0
        
        # 1) Parse consecutive runs
        runs = []
        current_char = word[0]
        count = 1
        
        for ch in word[1:]:
            if ch == current_char:
                count += 1
            else:
                runs.append(count)
                current_char = ch
                count = 1
        runs.append(count)
        
        # 2) Sum up (k_i - 1) for each run
        total_extra = sum((r - 1) for r in runs if r > 1)
        
        # 3) Add 1 for the "no error" case
        return total_extra + 1