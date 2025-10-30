class Solution:
    def countPairs(self, nums: List[int]) -> int:
        def remove_leading_zeros(s: str) -> str:
            # Remove leading zeros and return '0' if empty
            s = s.lstrip('0')
            return s if s else '0'
        
        def can_one_swap_transform(A: str, B: str) -> bool:
            # Check if A can be turned into B by at most one swap of digits in A
            # Compare as integers ignoring leading zeros
            if int(remove_leading_zeros(A)) == int(remove_leading_zeros(B)):
                return True
            
            A_list = list(A)
            for i in range(len(A_list)):
                for j in range(i + 1, len(A_list)):
                    # Swap digits at positions i and j
                    A_list[i], A_list[j] = A_list[j], A_list[i]
                    if int(remove_leading_zeros("".join(A_list))) == int(remove_leading_zeros(B)):
                        A_list[i], A_list[j] = A_list[j], A_list[i]  # swap back
                        return True
                    # swap back
                    A_list[i], A_list[j] = A_list[j], A_list[i]
            
            return False
        
        def is_almost_equal(x: int, y: int) -> bool:
            # Check if x and y are almost equal
            X, Y = str(x), str(y)
            # Either we can transform X into Y or Y into X by at most one swap
            return can_one_swap_transform(X, Y) or can_one_swap_transform(Y, X)
        
        count = 0
        n = len(nums)
        # count the number of pairs (i < j) that are almost equal
        for i in range(n):
            for j in range(i + 1, n):
                if is_almost_equal(nums[i], nums[j]):
                    count += 1
        
        return count