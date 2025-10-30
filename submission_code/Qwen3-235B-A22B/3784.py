from typing import List
from collections import deque, defaultdict

class Solution:
    def longestCommonPrefix(self, words: List[str], k: int) -> List[int]:
        n = len(words)
        if n < k:
            return [0] * n
        
        # Precompute the sorted array and their LCP array
        sorted_words = sorted(words)
        lcp = []
        for i in range(n - 1):
            a, b = sorted_words[i], sorted_words[i + 1]
            min_len = min(len(a), len(b))
            cnt = 0
            while cnt < min_len and a[cnt] == b[cnt]:
                cnt += 1
            lcp.append(cnt)
        
        # Precompute sliding window minima for the original lcp array with window size k-1
        window_size = k - 1
        if window_size == 0:
            # k == 1, handled separately
            max_len = max(len(word) for word in words)
            return [max_len if (n - 1) >= k else 0 for _ in range(n)]
        
        # Compute the sliding window minima using deque
        original_maxima = []
        dq = deque()
        for i in range(len(lcp) + 1):
            if i < window_size:
                continue
            while dq and dq[0] < i - window_size:
                dq.popleft()
            # Compute window from i - window_size to i -1
            # Wait, no, for window of size window_size (k-1), indices 0 to window_size-1 for first window
            # Adjust the loop accordingly
        dq = deque()
        sliding_min = []
        for i in range(len(lcp)):
            while dq and lcp[dq[-1]] >= lcp[i]:
                dq.pop()
            dq.append(i)
            if i >= window_size - 1:
                sliding_min.append(lcp[dq[0]])
            else:
                sliding_min.append(0)  # Placeholder, but not used later
        
        if not sliding_min:
            original_max = 0
        else:
            original_max = max(sliding_min) if len(lcp) >= window_size else 0
        
        # Now, process each word in the original array
        answer = []
        word_freq_global = defaultdict(int)
        for word in words:
            word_freq_global[word] += 1
        
        max_freq_len_global = 0
        freq_word_lengths = defaultdict(int)
        for word, freq in word_freq_global.items():
            if freq >= k:
                freq_word_lengths[len(word)] += 1
                if len(word) > max_freq_len_global:
                    max_freq_len_global = len(word)
        # If no such length, max_freq_len_global remains 0
        
        # Build a list of sorted words and their indices to find pos quickly
        # Since there may be duplicates, we need to track all positions
        # To handle this, build a list of tuples (word, index_in_original) and sort them
        indexed_words = [(word, idx) for idx, word in enumerate(words)]
        indexed_words_sorted = sorted(indexed_words)
        pos_map = [[] for _ in range(n)]
        # For each original index, store its position in the sorted array
        sorted_positions = [0] * n
        for sorted_idx, (word, orig_idx) in enumerate(indexed_words_sorted):
            pos_map[orig_idx] = sorted_idx
        
        # Precompute prefix counts for freq_max
        # Now, process each i in 0..n-1
        answer = [0] * n
        for orig_idx in range(n):
            m = n - 1
            if m < k:
                answer[orig_idx] = 0
                continue
            # Find the frequency-based max after removing orig_idx's word
            removed_word = words[orig_idx]
            # Temporarily adjust the frequency map
            word_freq = defaultdict(int)
            for word in words:
                if word == removed_word:
                    continue
                word_freq[word] += 1
            # Compute freq_max
            freq_max = 0
            for word, freq in word_freq.items():
                if freq >= k:
                    if len(word) > freq_max:
                        freq_max = len(word)
            # Now compute window_max in the modified sorted array
            # Compute the modified sorted array's LCP array
            modified_words = []
            for word in words:
                if word == removed_word:
                    continue
                modified_words.append(word)
            modified_words_sorted = sorted(modified_words)
            m_len = len(modified_words_sorted)
            if m_len < k:
                answer[orig_idx] = 0
                continue
            if k == 1:
                answer[orig_idx] = max(len(word) for word in modified_words_sorted)
                continue
            # Compute LCP array for modified_words_sorted
            lcp_mod = []
            for i in range(m_len - 1):
                a, b = modified_words_sorted[i], modified_words_sorted[i + 1]
                min_len = min(len(a), len(b))
                cnt = 0
                while cnt < min_len and a[cnt] == b[cnt]:
                    cnt += 1
                lcp_mod.append(cnt)
            # Compute sliding window minima for window_size = k-1
            window_size_mod = k - 1
            if window_size_mod > len(lcp_mod):
                window_max_mod = 0
            else:
                sliding_min_mod = []
                dq_mod = deque()
                for i in range(len(lcp_mod)):
                    # Remove elements out of the current window
                    while dq_mod and dq_mod[0] < i - window_size_mod + 1:
                        dq_mod.popleft()
                    # Remove elements larger than current element
                    while dq_mod and lcp_mod[dq_mod[-1]] >= lcp_mod[i]:
                        dq_mod.pop()
                    dq_mod.append(i)
                    # When window is formed
                    if i >= window_size_mod - 1:
                        sliding_min_mod.append(lcp_mod[dq_mod[0]])
                if sliding_min_mod:
                    window_max_mod = max(sliding_min_mod)
                else:
                    window_max_mod = 0
            answer[orig_idx] = max(freq_max, window_max_mod)
        
        return answer