class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        from collections import Counter, defaultdict
        
        count = Counter(nums)
        max_length = 1  # At least we can always pick one element
        
        for x in count:
            # Try different values of m, where k = 2^m
            m = 0
            while True:
                # Check if we can form the pattern with k = 2^m
                can_form = True
                total_length = 0
                
                if m == 0:
                    # k = 1, pattern is [x]
                    if count[x] >= 1:
                        total_length = 1
                    else:
                        can_form = False
                else:
                    # k = 2^m, pattern is [x, x^2, x^4, ..., x^(2^m), ..., x^4, x^2, x]
                    # Collect the requirements for each distinct value
                    requirements = defaultdict(int)
                    
                    for i in range(m):
                        power = x ** (2 ** i)
                        if power > 10**9:  # Constraint: nums[i] <= 10^9
                            can_form = False
                            break
                        requirements[power] += 2
                    
                    if can_form:
                        peak_power = x ** (2 ** m)
                        if peak_power > 10**9:
                            can_form = False
                        else:
                            requirements[peak_power] += 1
                    
                    # Check if we have enough copies for each required value
                    if can_form:
                        for value, needed in requirements.items():
                            if count[value] < needed:
                                can_form = False
                                break
                            total_length += needed
                
                if can_form:
                    max_length = max(max_length, total_length)
                    m += 1
                else:
                    break
        
        return max_length