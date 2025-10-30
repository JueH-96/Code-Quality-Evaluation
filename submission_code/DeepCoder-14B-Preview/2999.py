class Solution:
    def canBeEqual(self, s1: str, s2: str) -> bool:
        # Generate all possible transformations for s1
        s1_transforms = set()
        for swap0 in [False, True]:
            for swap1 in [False, True]:
                temp = list(s1)
                if swap0:
                    # Swap indices 0 and 2
                    temp[0], temp[2] = temp[2], temp[0]
                if swap1:
                    # Swap indices 1 and 3
                    temp[1], temp[3] = temp[3], temp[1]
                transformed = ''.join(temp)
                s1_transforms.add(transformed)
        
        # Generate all possible transformations for s2
        s2_transforms = set()
        for swap0 in [False, True]:
            for swap1 in [False, True]:
                temp = list(s2)
                if swap0:
                    # Swap indices 0 and 2
                    temp[0], temp[2] = temp[2], temp[0]
                if swap1:
                    # Swap indices 1 and 3
                    temp[1], temp[3] = temp[3], temp[1]
                transformed = ''.join(temp)
                s2_transforms.add(transformed)
        
        # Check if there is any common transformation
        return len(s1_transforms & s2_transforms) > 0