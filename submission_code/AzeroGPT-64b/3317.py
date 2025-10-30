from collections import Counter
from heapq import heappush, heappop

class Solution:
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        letterFreq = Counter()
        oddCounts = Counter()
        for word in words:
            wordCount = Counter(word)
            for letter, count in wordCount.items():
                letterFreq[letter] += count

        # Add letter occurrences as -ve values to a min heap to keep track of highest counts (as a max heap)
        maxOddCountHeap = []
        for count in letterFreq.values():
            if count % 2 != 0:
                oddCounts[letter] += 1
                heappush(maxOddCountHeap, -abs(count))
            elif count > 0:
                heappush(maxOddCountHeap, -abs(count))

        palindromesCount = 0
        while len(maxOddCountHeap) > 1:
            # If there is more than one odd count, we can reduce both by one and make a palindrome
            if oddCounts and maxOddCountHeap[0] < 0 and maxOddCountHeap[1] < 0:
                oddCounts -= 1
                palindromesCount += 1
                heappop(maxOddCountHeap)
                heappop(maxOddCountHeap)
            else:
                # If there is only one odd count, reduce it by two if possible
                # This is because a palindrome can have at most one odd-letter
                heappush(maxOddCountHeap, -(abs(heappop(maxOddCountHeap)) - 2))

            if len(maxOddCountHeap) == 1 and maxOddCountHeap[0] >= -2:
                # If only one element remains and its count is >= -2 (was odd or exactly 2),
                # then exactly one palindrome can be made now.
                palindromesCount += 1
                maxOddCountHeap.clear()

        return palindromesCount + len(words) - len(oddCounts)