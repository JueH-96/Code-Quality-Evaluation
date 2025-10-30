class Solution:
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        events = []
        for num in nums:
            events.append((num - k, 'start'))
            events.append((num + k, 'end'))

        events.sort(key=lambda x: (x[0], -1 if x[1] == 'start' else 1))

        max_beauty = 0
        current_beauty = 0

        for _, type in events:
            if type == 'start':
                current_beauty += 1
            else:
                current_beauty -= 1
            max_beauty = max(max_beauty, current_beauty)

        return max_beauty