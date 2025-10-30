class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        wordsContainer.sort(key=len, reverse=True)
        wordsQuery.sort(key=len, reverse=True)
        result = []
        for query in wordsQuery:
            l = 0
            r = len(wordsContainer) - 1
            while l <= r:
                mid = (l + r) // 2
                if wordsContainer[mid] == query:
                    result.append(mid)
                    break
                elif wordsContainer[mid] < query:
                    l = mid + 1
                else:
                    r = mid - 1
        return result