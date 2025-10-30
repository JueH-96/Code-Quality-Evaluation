class Solution:
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        # 1. Count the total frequency of all characters across all words.
        freq = [0]*26
        for w in words:
            for ch in w:
                freq[ord(ch) - ord('a')] += 1
        
        # 2. Compute how many pairs and single leftovers we have globally.
        pairLeft = sum(f // 2 for f in freq)
        singleLeft = sum(f % 2 for f in freq)
        
        # 3. For each word, record (pairs_needed, is_odd_length).
        #    pairs_needed = length // 2, is_odd_length = length % 2
        lengths = []
        for w in words:
            L = len(w)
            lengths.append((L // 2, L % 2))
        
        # 4. Sort by (pairs_needed ASC, is_odd_length ASC).
        #    This ensures we use fewer resources first and prefer even-length
        #    palindromes among those needing the same number of pairs.
        lengths.sort(key=lambda x: (x[0], x[1]))
        
        ans = 0
        # 5. Greedily try to form palindromes from smallest "cost" to largest.
        for cost, is_odd in lengths:
            neededPairs = cost
            neededSingles = is_odd
            # If we need a single and have none available, we must break 1 pair -> "additional = 1"
            additional = 1 if (neededSingles == 1 and singleLeft == 0) else 0
            totalPairCost = neededPairs + additional
            
            # Check if we have enough pairs left to pay that cost.
            if pairLeft >= totalPairCost:
                # Pay the pair cost.
                pairLeft -= totalPairCost
                # If we broke a pair, we gain 2 singles.
                if additional == 1:
                    singleLeft += 2
                # Now use up one single if this word length is odd.
                singleLeft -= neededSingles
                ans += 1
        
        return ans