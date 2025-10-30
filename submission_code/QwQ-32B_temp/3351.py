class Solution:
    def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
        happiness.sort(reverse=True)
        total = 0
        for i in range(k):
            current = happiness[i] - i
            if current > 0:
                total += current
            else:
                # Once current is non-positive, further elements will also be non-positive since the list is sorted
                # So we can break early here
                break
        return total