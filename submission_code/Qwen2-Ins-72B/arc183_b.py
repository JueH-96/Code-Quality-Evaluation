from collections import defaultdict
from typing import List

class ArrayModifier:
    """
    Determines if it's possible to make two arrays A and B identical by performing a specific operation.
    The operation allows changing the value of A[i] to A[j] where |i - j| <= K.
    """
    
    def canMakeIdentical(self, A: List[int], B: List[int], K: int) -> bool:
        """
        Checks if it's possible to make array A identical to array B.
        
        Parameters:
        A (List[int]): The initial array.
        B (List[int]): The target array.
        K (int): The maximum distance between indices for the operation.
        
        Returns:
        bool: True if it's possible to make A identical to B, False otherwise.
        """
        if A == B:
            return True
        
        # Create a dictionary to store the last seen index of each value in A
        last_seen = defaultdict(int)
        for i, val in enumerate(A):
            last_seen[val] = i
        
        # Check if each value in B can be reached from the corresponding value in A
        for i, val in enumerate(B):
            if val not in last_seen:
                return False
            if abs(i - last_seen[val]) > K:
                return False
        
        return True

def main():
    T = int(input())
    solver = ArrayModifier()
    for _ in range(T):
        N, K = map(int, input().split())
        A = list(map(int, input().split()))
        B = list(map(int, input().split()))
        if solver.canMakeIdentical(A, B, K):
            print("Yes")
        else:
            print("No")

if __name__ == "__main__":
    main()