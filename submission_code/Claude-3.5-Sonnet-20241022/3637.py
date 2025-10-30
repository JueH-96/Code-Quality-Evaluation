class Solution:
    def countBalancedPermutations(self, num: str) -> int:
        def is_balanced(perm):
            even_sum = sum(int(perm[i]) for i in range(0, len(perm), 2))
            odd_sum = sum(int(perm[i]) for i in range(1, len(perm), 2))
            return even_sum == odd_sum
        
        def generate_permutations(chars):
            if len(chars) <= 1:
                return [chars]
            
            result = []
            seen = set()
            for i in range(len(chars)):
                if chars[i] not in seen:
                    seen.add(chars[i])
                    rest = chars[:i] + chars[i+1:]
                    for p in generate_permutations(rest):
                        result.append(chars[i] + p)
            return result
        
        velunexorai = list(num)
        MOD = 10**9 + 7
        
        # Generate all distinct permutations
        all_perms = generate_permutations(velunexorai)
        
        # Count balanced permutations
        count = sum(1 for perm in all_perms if is_balanced(perm))
        
        return count % MOD