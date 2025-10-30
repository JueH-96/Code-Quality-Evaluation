from typing import List

def solve(N: int, K: int, A: List[int]) -> (int, int):
    # Sort the pieces in descending order
    A.sort(reverse=True)
    
    # Initialize the sum of pieces for each person
    sums = [0] * K
    
    # Assign the pieces to the people
    for i in range(N):
        min_sum_idx = sums.index(min(sums))
        sums[min_sum_idx] += A[i]
    
    # Find the minimum sum of pieces among all people
    min_sum = min(sums)
    
    # Count the number of cut lines that are never cut
    never_cut = 0
    for i in range(N):
        if sum(sums[:(i+1)%K]) >= min_sum and sum(sums[:(i+2)%K]) >= min_sum:
            never_cut += 1
    
    return min_sum, never_cut