import heapq
from typing import List

class Solution:
  """
  Finds the maximum sum of at most k elements from a 2D grid, subject to row limits.
  Uses a greedy approach by selecting the overall largest elements that satisfy constraints.
  """
  def maxSum(self, grid: List[List[int]], limits: List[int], k: int) -> int:
    """
    Args:
        grid: A list of lists of integers representing the 2D matrix. n x m size.
              Constraints: 1 <= n, m <= 500. 0 <= grid[i][j] <= 10^5.
        limits: A list of integers where limits[i] is the maximum number of elements
                that can be taken from row i. Length n. 
                Constraints: 0 <= limits[i] <= m. n == len(limits).
        k: The maximum total number of elements to be taken from the grid.
           Constraints: 0 <= k <= min(n * m, sum(limits)).

    Returns:
        The maximum possible sum.

    The strategy is based on the observation that to maximize the sum, we should always
    prioritize picking larger elements. The constraints limit which elements are available.
    From row `i`, only the `limits[i]` largest elements are potentially selectable.
    We gather all such potentially selectable elements from all rows into a single pool.
    Then, from this pool, we select the top `k` largest elements. The sum of these `k`
    elements will be the maximum possible sum under the given constraints.
    """
    
    n = len(grid)
    # Constraints state n >= 1 and m >= 1, so grid and rows are non-empty.
    # No need to check for empty grid/rows explicitly.
    # m = len(grid[0]) # m is not explicitly needed in the logic below, but good to know

    # List to store all potentially selectable elements based on row limits.
    # An element grid[i][j] is potentially selectable if it's among the 'limits[i]' largest elements in row i.
    all_potential_gains = []
    
    for i in range(n):
        # Sort row i in descending order to easily identify the largest elements.
        # Using sorted() creates a new sorted list and does not modify the input grid.
        row_sorted = sorted(grid[i], reverse=True)
        
        # Add the values of the top 'limits[i]' elements from this row to the pool.
        # The constraint 0 <= limits[i] <= m ensures we access valid indices 
        # (up to m elements in row_sorted) and respect the per-row limit.
        # If limits[i] is 0, range(0) is empty, and nothing is added for this row, which is correct.
        for j in range(limits[i]):
             # Append the value of the (j+1)-th largest element in row i
            all_potential_gains.append(row_sorted[j])
    
    # If k is 0, we cannot select any elements, so the maximum sum is 0.
    if k == 0:
        return 0
        
    # Use heapq.nlargest to efficiently find the k largest elements among all potential gains.
    # `heapq.nlargest(k, iterable)` finds the k largest items in the iterable.
    # It returns a list of these k items. If the iterable has fewer than k items, it returns all items.
    # The problem constraint `k <= sum(limits)` means k is less than or equal to the total number 
    # of elements added to `all_potential_gains`. Let S = sum(limits). Then len(all_potential_gains) = S.
    # Since k <= S, `heapq.nlargest` will return exactly k elements if k > 0 and S > 0.
    # If S = 0, then all limits[i] must be 0. The constraint k <= S implies k=0. 
    # The k=0 case is handled above. So if we reach here, k > 0 and S >= k > 0.
    top_k_gains = heapq.nlargest(k, all_potential_gains)
    
    # The maximum sum is the sum of these k largest chosen elements.
    # sum() works on the list returned by heapq.nlargest.
    max_sum = sum(top_k_gains)
            
    return max_sum