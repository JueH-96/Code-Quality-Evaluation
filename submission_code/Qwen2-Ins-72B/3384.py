class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        n = len(word)
        groups = defaultdict(list)
        
        for i, c in enumerate(word):
            groups[i % k].append(c)
        
        ans = 0
        for group in groups.values():
            counter = Counter(group)
            most_common, count = counter.most_common(1)[0]
            ans += len(group) - count
        
        return ans