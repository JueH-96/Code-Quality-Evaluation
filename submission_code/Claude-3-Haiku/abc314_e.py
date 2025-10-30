import math
from typing import List

def roulette_expected_cost(N: int, M: int, C: List[int], P: List[int], S: List[List[int]]) -> float:
    """
    Calculates the expected cost for Takahashi to earn at least M points by playing the roulette wheels.
    """
    # Create a list of tuples (cost, points) for each wheel
    wheels = [(C[i], [S[i][j] for j in range(P[i])]) for i in range(N)]
    
    # Define a function to calculate the expected cost for a given target score
    def expected_cost(target: int) -> float:
        # Initialize the expected cost to 0
        expected = 0.0
        
        # Iterate over the wheels
        for cost, points in wheels:
            # Calculate the probability of earning at least the target score on this wheel
            prob = sum(1 for p in points if p >= target) / P[i]
            
            # Add the expected cost for this wheel to the total
            expected += cost / prob
        
        return expected
    
    # Use binary search to find the minimum expected cost
    left, right = 1, M
    while left <= right:
        mid = (left + right) // 2
        if expected_cost(mid) <= expected_cost(M):
            right = mid - 1
        else:
            left = mid + 1
    
    return expected_cost(M)