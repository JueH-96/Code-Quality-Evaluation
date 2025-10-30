class Solution:
    def doesValidArrayExist(self, derived: List[int]) -> bool:
        n = len(derived)
        
        #Check for invalid cases
        if n == 0:
            return True
        if n == 1 and derived[0] != 0:
            return False
        if n % 2 != 0 and sum(derived) % 2 != 0:
            return False

        
        for i in range(2):
            original = [i]
            valid = True
            for j in range(n - 1):
                next_val = original[-1] ^ derived[j]
                original.append(next_val)
            if (original[-1] ^ original[0]) != derived[-1]:
                valid = False

            if valid:
                return True
        return False