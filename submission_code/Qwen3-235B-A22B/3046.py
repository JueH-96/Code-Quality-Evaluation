class Solution:
    def minimumOperations(self, num: str) -> int:
        n = len(num)
        candidates = []
        endings = ["00", "25", "50", "75"]
        
        for ending in endings:
            first, second = ending[0], ending[1]
            found_j = -1
            # Find the rightmost j where num[j] == second
            for j in reversed(range(n)):
                if num[j] == second:
                    found_j = j
                    # Find the rightmost i < found_j where num[i] == first
                    for i in reversed(range(found_j)):
                        if num[i] == first:
                            current_deletions = n - (i + 2)
                            candidates.append(current_deletions)
                            break  # Break inner loop to take the rightmost i
                    break  # Break outer loop after finding the rightmost pair
        
        # Check for the zero case (delete all except one '0') if possible
        has_zero = any(c == '0' for c in num)
        if has_zero:
            candidates.append(n - 1)
        
        # Always consider the case of deleting all digits (all_delete_case)
        candidates.append(n)
        
        return min(candidates)