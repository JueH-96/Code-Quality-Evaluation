class Solution:
    def maxDistinctElements(self, nums: List[int], k: int) -> int:
        counts = {}
        for num in nums:
            counts[num] = counts.get(num, 0) + 1

        distinct_count = len(counts)
        duplicates = []
        for num, count in counts.items():
            if count > 1:
                duplicates.append(num)

        duplicates.sort()

        used = set(counts.keys())

        for num in duplicates:
            count = counts[num]
            for _ in range(count - 1):
                found = False
                for offset in range(1, k + 1):
                    target = num + offset
                    if target not in used:
                        used.add(target)
                        distinct_count += 1
                        k -= offset
                        found = True
                        break
                if found:
                    continue
                for offset in range(1, k + 1):
                    target = num - offset
                    if target not in used:
                        used.add(target)
                        distinct_count += 1
                        k -= offset
                        found = True
                        break
                if not found:
                    break # Cannot make this duplicate distinct with remaining k
                if k <= 0:
                    break
            if k <= 0:
                break

        return distinct_count