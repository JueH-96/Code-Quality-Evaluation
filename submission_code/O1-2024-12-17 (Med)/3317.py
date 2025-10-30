class Solution:
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        from collections import Counter
        
        # 1) Count total character frequencies
        freq = Counter()
        for w in words:
            freq.update(w)
        
        # 2) Compute total number of "pairs" (P) we can form and how many single leftovers (S) we have
        #    P = sum over all characters freq[c] // 2
        #    S = sum over all characters freq[c] % 2
        P = sum(v // 2 for v in freq.values())
        S = sum(v % 2 for v in freq.values())
        
        # 3) Separate the words into even-length and odd-length, computing their "pair cost":
        #    For an even-length L, cost = L//2 (needs that many pairs, 0 singles)
        #    For an odd-length L, cost = L//2 (needs that many pairs, plus 1 leftover single)
        even_costs = []
        odd_costs = []
        for w in words:
            length = len(w)
            if length % 2 == 0:
                # even length
                even_costs.append(length // 2)
            else:
                # odd length
                odd_costs.append(length // 2)  # plus 1 single leftover if we use it
        
        # 4) Sort costs ascending
        even_costs.sort()
        odd_costs.sort()
        
        # 5) Merge them into one list of (cost, isOdd), sorted by cost ascending
        merged = []
        for c in even_costs:
            merged.append((c, 0))
        for c in odd_costs:
            merged.append((c, 1))
        merged.sort(key=lambda x: x[0])  # sort by the pair-cost
        
        # 6) Greedily pick from smallest to largest cost,
        #    subject to not exceeding total pairs P and not exceeding S odd picks
        used_pairs = 0
        used_odd_count = 0
        total_palindromes = 0
        
        for cost, is_odd in merged:
            if used_pairs + cost <= P:
                if is_odd == 1:
                    # need one leftover single to form an odd palindrome
                    if used_odd_count < S:
                        used_odd_count += 1
                        used_pairs += cost
                        total_palindromes += 1
                    else:
                        # no more singles left to use for an odd palindrome
                        continue
                else:
                    # even palindrome only needs pairs
                    used_pairs += cost
                    total_palindromes += 1
            else:
                # not enough pair budget left
                continue
        
        return total_palindromes