Wait, I think I'm overcomplicating this. Let me reconsider the problem more carefully.

When we perform a front kick in a direction, cells up to 2 steps away in that direction become roads. Then we can move freely on all connected roads.

A simpler approach:
1. Use 0-1 BFS where moving to adjacent roads costs 0
2. When at a position, we can perform a kick in any direction and then move to any cell that becomes reachable (cost 1)