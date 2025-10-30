from typing import List

class Solution:
  def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
    cuts = []
    
    # Combine all cuts into a single list with their costs and types
    # We can use a simple tuple (cost, type_char) for this
    for cost in horizontalCut:
      cuts.append((cost, 'H')) # 'H' for horizontal
    for cost in verticalCut:
      cuts.append((cost, 'V')) # 'V' for vertical
    
    # Sort cuts by cost in descending order
    # This is the core of the greedy strategy: process more expensive cuts first
    # to ensure their multipliers are small.
    cuts.sort(key=lambda x: x[0], reverse=True)
    
    # Initialize piece counts.
    # horizontal_pieces: number of segments a vertical cut line would cross.
    # vertical_pieces: number of segments a horizontal cut line would cross.
    # Initially, the cake is one whole piece in both dimensions.
    horizontal_pieces = 1 
    vertical_pieces = 1   
    total_cost = 0
    
    # Iterate through sorted cuts
    for cost, cut_type in cuts:
      if cut_type == 'H':
        # This horizontal cut spans across 'vertical_pieces' segments
        total_cost += cost * vertical_pieces
        # This cut adds one more horizontal piece
        horizontal_pieces += 1
      else: # cut_type == 'V'
        # This vertical cut spans across 'horizontal_pieces' segments
        total_cost += cost * horizontal_pieces
        # This cut adds one more vertical piece
        vertical_pieces += 1
        
    return total_cost