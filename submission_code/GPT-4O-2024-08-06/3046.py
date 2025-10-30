class Solution:
    def minimumOperations(self, num: str) -> int:
        # Possible endings for a number to be divisible by 25
        endings = ["00", "25", "50", "75"]
        n = len(num)
        min_operations = float('inf')
        
        # Check for each possible ending
        for ending in endings:
            # Start from the end of the string and try to match the ending
            j = len(ending) - 1
            # We will count how many characters we need to delete
            delete_count = 0
            
            # Traverse the string from the end
            for i in range(n - 1, -1, -1):
                if num[i] == ending[j]:
                    j -= 1
                    if j < 0:
                        # If we matched the entire ending, calculate deletions
                        min_operations = min(min_operations, delete_count)
                        break
                else:
                    delete_count += 1
        
        # If no ending matched, we need to delete all but one digit to make it 0
        if min_operations == float('inf'):
            min_operations = n - 1
        
        return min_operations