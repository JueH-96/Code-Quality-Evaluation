import math
from typing import List

class Solution:
  """
  Calculates the minimum cost to cut an m x n cake into 1 x 1 pieces.
  """
  def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
    """
    Finds the minimum cost using a greedy approach by prioritizing cuts with higher costs.

    The core idea is that the cost of making a cut depends on how many pieces it divides.
    A horizontal cut divides pieces along its path. The number of pieces it divides is equal to the
    number of vertical segments the cake currently has.
    Similarly, a vertical cut divides pieces along its path. The number of pieces it divides is equal
    to the number of horizontal segments the cake currently has.
    
    To minimize the total cost, we should prioritize making cuts with higher base costs earlier,
    when the number of segments they pass through (and thus the multiplier for their cost) is smaller.
    Making a vertical cut increases the number of vertical segments, potentially increasing the cost of subsequent horizontal cuts.
    Making a horizontal cut increases the number of horizontal segments, potentially increasing the cost of subsequent vertical cuts.
    Therefore, a greedy strategy of always performing the available cut with the highest base cost is optimal.

    Args:
      m: The number of rows in the cake (height).
      n: The number of columns in the cake (width).
      horizontalCut: A list where horizontalCut[i] is the cost to cut along horizontal line i. Length m-1.
      verticalCut: A list where verticalCut[j] is the cost to cut along vertical line j. Length n-1.

    Returns:
      The minimum total cost to cut the entire cake into 1x1 pieces.
    """

    # Create a list of all cuts represented as tuples (cost, type).
    # 'H' denotes a horizontal cut, 'V' denotes a vertical cut.
    cuts = []
    # Add all horizontal cuts with their costs.
    # There are m-1 horizontal cuts possible.
    for i in range(m - 1):
        cuts.append((horizontalCut[i], 'H'))
    # Add all vertical cuts with their costs.
    # There are n-1 vertical cuts possible.
    for i in range(n - 1):
        cuts.append((verticalCut[i], 'V'))
        
    # Sort the cuts in descending order based on cost. The greedy strategy is to perform
    # the cuts with the highest costs first. This minimizes the multiplier applied to
    # these expensive cuts. When costs are equal, the relative order of cuts with the same cost
    # does not impact the total minimum cost, as shown by analysis.
    cuts.sort(key=lambda x: x[0], reverse=True)
    
    total_cost = 0
    # Initialize the number of horizontal and vertical segments.
    # The cake starts as a single piece, which means 1 horizontal segment and 1 vertical segment.
    horizontal_segments = 1 
    vertical_segments = 1   
    
    # Process each cut in the sorted order (highest cost first).
    for cost, type in cuts:
        if type == 'H':
            # If the cut is horizontal:
            # The cost incurred is the cut's base cost multiplied by the current number of vertical segments
            # because the cut passes through all existing vertical segments.
            total_cost += vertical_segments * cost
            # Performing a horizontal cut increases the number of horizontal segments by 1.
            # This will affect the cost calculations for future vertical cuts.
            horizontal_segments += 1
        else: # type == 'V'
            # If the cut is vertical:
            # The cost incurred is the cut's base cost multiplied by the current number of horizontal segments
            # because the cut passes through all existing horizontal segments.
            total_cost += horizontal_segments * cost
            # Performing a vertical cut increases the number of vertical segments by 1.
            # This will affect the cost calculations for future horizontal cuts.
            vertical_segments += 1
            
    # After processing all necessary cuts (m-1 horizontal and n-1 vertical), 
    # the total accumulated cost is the minimum possible cost.
    return total_cost