class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        import heapq

        # Dictionary to keep track of the current frequency of each ID
        current_freq = {}
        # A max-heap (we'll store negative frequencies to simulate a max heap using Python's min heap)
        max_heap = []
        # Result list
        ans = []

        for i in range(len(nums)):
            # Update the frequency of the current ID
            old_count = current_freq.get(nums[i], 0)
            new_count = old_count + freq[i]

            # If new_count is 0, remove it from the dictionary (collection) to keep things clean
            if new_count == 0:
                if nums[i] in current_freq:
                    del current_freq[nums[i]]
            else:
                current_freq[nums[i]] = new_count
                # Push the new frequency (negative for max-heap) along with the ID onto the heap
                heapq.heappush(max_heap, (-new_count, nums[i]))

            # Pop stale entries from the heap top: those whose frequency no longer matches current_freq
            while max_heap:
                neg_freq_top, id_top = max_heap[0]
                # The actual frequency is -neg_freq_top
                # Check if it matches what's in current_freq
                if current_freq.get(id_top, 0) == -neg_freq_top:
                    # It's valid, so break
                    break
                else:
                    # It's stale, pop it off
                    heapq.heappop(max_heap)

            # Now, if the heap is empty, the collection is empty
            if not max_heap:
                ans.append(0)
            else:
                # The heap top's frequency is the most frequent count
                ans.append(-max_heap[0][0])

        return ans