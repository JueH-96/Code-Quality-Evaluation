class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        """
        We want to find the maximum number of partitions after:
         1) Changing at most one character in s to any (possibly new) lowercase letter,
         2) Then repeatedly taking the LONGEST prefix with at most k distinct letters,
            removing it, and counting one partition for each removal.
        
        A direct (and correct) way is:
         - Compute how many partitions we get with no change.
         - For each index i and each possible changed letter c != s[i],
           compute how many partitions would result and take the maximum.
        
        However, naively this can be up to 26*n times an O(n) "partition check",
        i.e. 26*n^2. For n up to 10^4, that can be too large in Python.

        In practice, we implement this carefully with:
         - A fast O(n) routine to compute the partition count if exactly one character
           (at index i) is changed to letter c.
         - Pruning/early-stopping if we cannot possibly exceed the best result so far.
         - Only 25 candidates for c at each i.

        This still risks high worst-case cost in Python, but with careful implementation
        and pruning it can often pass in contest-style environments. If there were tighter
        time limits, a more sophisticated approach or a lower-level language might be needed.

        The key partitioning routine (largest prefix with ≤ k distinct) is straightforward:
          freq array counts frequencies of letters in the current chunk;
          once distinct-count exceeds k, we end a partition and reset.

        We'll do:
          1) Compute the "no-change" count.
          2) Try each position i, and each candidate letter c != s[i], compute the partition
             count. Keep track of the global maximum.
          3) Return that maximum.
        """

        import sys
        input_str = s  # since we already have s

        n = len(input_str)
        arr = [ord(ch) - ord('a') for ch in input_str]  # convert string to list of ints [0..25]

        # Fast routine to compute partition count with NO change
        def partition_count_no_change(k):
            freq = [0]*26
            distinct = 0
            partitions = 0
            for c in arr:
                if freq[c] == 0:
                    distinct += 1
                freq[c] += 1
                if distinct > k:
                    # start a new partition
                    partitions += 1
                    freq = [0]*26
                    freq[c] = 1
                    distinct = 1
            return partitions + 1

        # Fast routine to compute the partition count when exactly one character
        # at index idx is changed to new_c. We also do early pruning if we can't beat best_so_far.
        def partition_count_with_one_change(k, idx, new_c, best_so_far):
            freq = [0]*26
            distinct = 0
            partitions = 0
            for j in range(n):
                c = new_c if j == idx else arr[j]
                if freq[c] == 0:
                    distinct += 1
                freq[c] += 1
                if distinct > k:
                    partitions += 1
                    freq = [0]*26
                    freq[c] = 1
                    distinct = 1

                # Prune if the maximum possible partitions (each remaining char forms its own partition)
                # cannot exceed best_so_far
                # Max possible if from j+1..n-1 we make each character a partition => (n-1 - j)
                # plus (partitions so far)
                if partitions + (n - 1 - j) <= best_so_far:
                    return partitions  # no need to continue, can't beat best_so_far
            return partitions + 1

        # 1) Base answer: no change
        base = partition_count_no_change(k)
        best = base

        # If best already equals n, we can't do better than that (each char is its own partition).
        if best == n:
            return best

        # 2) Try changing each index i to each of 25 other letters:
        #    We do an early break in the inner loop if best reaches n.
        for i in range(n):
            orig_char = arr[i]
            for c in range(26):
                if c == orig_char:
                    continue
                # Compute partitions if we change s[i] to letter c
                candidate = partition_count_with_one_change(k, i, c, best)
                if candidate > best:
                    best = candidate
                    if best == n:  # can't do better than n
                        return n
            # A small optional pruning if best == n:
            if best == n:
                return n

        return best