class Solution:
    def shiftDistance(self, s: str, t: str, nextCost: List[int], previousCost: List[int]) -> int:
        # Precompute the cost of shifting each letter forward and backward by d steps (0 <= d <= 26).
        # forwardCost[x][d] = sum of nextCosts if we shift letter x forward by d steps.
        # backwardCost[x][d] = sum of previousCosts if we shift letter x backward by d steps.
        
        # We will build two 2D arrays: 
        #   forwardCost[x][d] holds the cost of going from x to x+d (mod 26) by repeated forward shifts.
        #   backwardCost[x][d] holds the cost of going from x to x-d (mod 26) by repeated backward shifts.
        
        # Each row x has 27 columns for distances d = 0..26 (only 0..25 are really needed).
        # forwardCost[x][0] = 0, 
        # forwardCost[x][1] = nextCost[x], 
        # forwardCost[x][2] = nextCost[x] + nextCost[(x+1) mod 26], etc.
        
        ALPHABET_SIZE = 26
        forwardCost = [[0] * (ALPHABET_SIZE + 1) for _ in range(ALPHABET_SIZE)]
        backwardCost = [[0] * (ALPHABET_SIZE + 1) for _ in range(ALPHABET_SIZE)]
        
        # Build forwardCost
        for x in range(ALPHABET_SIZE):
            for d in range(1, ALPHABET_SIZE + 1):
                # The letter we are shifting from in this step is (x + d - 1) mod 26
                letter_from = (x + d - 1) % ALPHABET_SIZE
                forwardCost[x][d] = forwardCost[x][d - 1] + nextCost[letter_from]
        
        # Build backwardCost
        for x in range(ALPHABET_SIZE):
            for d in range(1, ALPHABET_SIZE + 1):
                # The letter we are shifting from in this step is (x - (d - 1)) mod 26
                letter_from = (x - (d - 1)) % ALPHABET_SIZE
                backwardCost[x][d] = backwardCost[x][d - 1] + previousCost[letter_from]
        
        total_cost = 0
        for i in range(len(s)):
            sc = ord(s[i]) - ord('a')  # source char index
            tc = ord(t[i]) - ord('a')  # target char index
            
            if sc == tc:
                continue  # cost = 0 if no change needed
            
            # distance forward (how many steps from sc to tc if we move forward)
            if tc >= sc:
                forward_dist = tc - sc
            else:
                forward_dist = tc + ALPHABET_SIZE - sc
            
            # distance backward (how many steps from sc to tc if we move backward)
            if sc >= tc:
                backward_dist = sc - tc
            else:
                backward_dist = sc + ALPHABET_SIZE - tc
            
            cost_forward = forwardCost[sc][forward_dist]
            cost_backward = backwardCost[sc][backward_dist]
            
            total_cost += min(cost_forward, cost_backward)
        
        return total_cost