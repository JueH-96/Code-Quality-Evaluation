class Solution:
    def findMinimumTime(self, strength: List[int], K: int) -> int:
        n = len(strength)
        import itertools
        
        def calculate_time(permutation):
            time = 0
            energy = 0
            x = 1
            
            for lock_index in permutation:
                required_energy = strength[lock_index]
                
                while energy < required_energy:
                    energy += x
                    time += 1
                
                energy = 0
                x += K
            
            return time
        
        min_time = float('inf')
        for permutation in itertools.permutations(range(n)):
            min_time = min(min_time, calculate_time(permutation))
        
        return min_time