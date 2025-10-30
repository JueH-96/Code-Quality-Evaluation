class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        n = len(s)
        
        def is_semi_palindrome(sub):
            m = len(sub)
            for d in range(1, m):
                if m % d == 0:
                    is_palindrome = True
                    for i in range(d):
                        temp = ""
                        for j in range(i, m, d):
                            temp += sub[j]
                        if temp != temp[::-1]:
                            is_palindrome = False
                            break
                    if is_palindrome:
                        return True
            return False

        def count_changes(sub):
            count = 0
            m = len(sub)
            for d in range(1, m):
                if m % d == 0:
                    min_changes_d = 0
                    for i in range(d):
                        temp = ""
                        for j in range(i, m, d):
                            temp += sub[j]
                        
                        freq = {}
                        for char in temp:
                            freq[char] = freq.get(char, 0) + 1
                        
                        max_freq = 0
                        for char in freq:
                            max_freq = max(max_freq, freq[char])
                        min_changes_d += len(temp) - max_freq

                    
                    if min_changes_d < count or count == 0:
                        count = min_changes_d
            return count if count >0 else len(sub) -1

        
        min_total_changes = float('inf')
        
        for i in range(1 << (n - 1)):
            partitions = []
            current_partition = ""
            for j in range(n):
                current_partition += s[j]
                if (i >> j) & 1 or j == n - 1:
                    partitions.append(current_partition)
                    current_partition = ""
            
            if len(partitions) != k:
                continue

            total_changes = 0
            for partition in partitions:
                total_changes += count_changes(partition)
            min_total_changes = min(min_total_changes, total_changes)

        return min_total_changes