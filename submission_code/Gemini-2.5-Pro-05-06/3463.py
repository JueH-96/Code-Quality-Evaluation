from typing import List

class Solution:
  def numberOfAlternatingGroups(self, colors: List[int]) -> int:
    n = len(colors)
    
    # Constraints: 3 <= colors.length <= 100.
    # This means n is always at least 3.

    alternating_groups_count = 0

    # Iterate through each possible starting tile of a 3-tile group.
    # There are n such possible starting positions in a circle of n tiles.
    for i in range(n):
      # Define the three contiguous tiles for the current window.
      # The window is (colors[i], colors[(i+1)%n], colors[(i+2)%n]).
      
      # Left tile of the triplet
      left_tile = colors[i] 
      # Note: colors[i] is equivalent to colors[i % n] here, 
      # because the loop `for i in range(n)` ensures `0 <= i < n`.
      
      # Middle tile of the triplet
      # The index (i + 1) needs modulo n for circular wrap-around.
      middle_tile_idx = (i + 1) % n
      middle_tile = colors[middle_tile_idx]
      
      # Right tile of the triplet
      # The index (i + 2) needs modulo n for circular wrap-around.
      right_tile_idx = (i + 2) % n
      right_tile = colors[right_tile_idx]

      # An alternating group is defined by "the middle tile has a different color 
      # from its left and right tiles".
      # This means: (middle_tile != left_tile) AND (middle_tile != right_tile)
      
      # Since colors[i] can only be 0 or 1 (binary colors):
      # - If middle_tile is 0, then left_tile must be 1 and right_tile must be 1.
      # - If middle_tile is 1, then left_tile must be 0 and right_tile must be 0.
      # In both scenarios, it implies that left_tile == right_tile.
      #
      # Therefore, for binary colors, the condition is equivalent to:
      #   (left_tile == right_tile) AND (left_tile != middle_tile)
      # We use this equivalent form.
      
      if left_tile == right_tile and left_tile != middle_tile:
        alternating_groups_count += 1
            
    return alternating_groups_count