Let me trace through the first example to verify:
- N=4, A=[5,7,0,3], exchanges=[(2,2),(4,3),(5,2)]
- multiplier[3] = 1.0
- multiplier[2] = 2 * 1.0 / 5 = 0.4
- multiplier[1] = 3 * 0.4 / 4 = 0.3  
- multiplier[0] = 2 * 0.3 / 2 = 0.3

This doesn't seem right. Let me try a different approach - dynamic programming working forward: