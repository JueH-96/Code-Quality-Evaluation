class Solution:
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        n = len(word)

        # Check if it's already in a self-cyclic state or impossible to revert.
        if n % k == 0 or k == 1:
            return 1
        
        # Find the minimum cycle time.
        time = 2
        while True:
            cyclic_k = k * time
            if (n * 2) % cyclic_k == 0:
                return time
            
            time += 1