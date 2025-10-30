class Solution:
    def maxDistance(self, s: str, k: int) -> int:
        
        # Define the score contributions for each of the four objective functions.
        # These maps specify how each character ('N', 'S', 'E', 'W') contributes
        # to the value of the objective function (e.g., x+y).
        # A contribution of +1 means it's a "good" move for this objective.
        # A contribution of -1 means it's a "bad" move for this objective,
        # and we might consider changing it if k_remaining > 0.
        
        score_maps = [
            # Type 0: Maximize x + y
            # N(0,1) -> 1, S(0,-1) -> -1, E(1,0) -> 1, W(-1,0) -> -1
            {'N': 1, 'S': -1, 'E': 1, 'W': -1},
            
            # Type 1: Maximize x - y
            # N(0,1) -> -1, S(0,-1) -> 1, E(1,0) -> 1, W(-1,0) -> -1
            {'N': -1, 'S': 1, 'E': 1, 'W': -1},
            
            # Type 2: Maximize -x + y
            # N(0,1) -> 1, S(0,-1) -> -1, E(1,0) -> -1, W(-1,0) -> 1
            {'N': 1, 'S': -1, 'E': -1, 'W': 1},
            
            # Type 3: Maximize -x - y
            # N(0,1) -> -1, S(0,-1) -> 1, E(1,0) -> -1, W(-1,0) -> 1
            {'N': -1, 'S': 1, 'E': -1, 'W': 1}
        ]

        overall_max_dist = 0

        # Run four separate simulations, one for each objective function.
        # The Manhattan distance |x| + |y| is equal to max(x+y, x-y, -x+y, -x-y).
        # By maximizing each of these four linear combinations independently and
        # taking the maximum of their results, we find the overall maximum Manhattan distance.
        for score_map in score_maps:
            current_score = 0  # Tracks the current value of the linear combination (e.g., x+y)
            k_remaining = k    # Number of changes remaining for this simulation
            # Tracks the maximum value of the current linear combination achieved over any prefix.
            # Initialized to 0 because starting at (0,0) yields a score of 0.
            max_score_for_this_type = 0 

            for char_move in s:
                original_contribution = score_map[char_move]

                if original_contribution == 1:
                    # This is a "good" move for the current objective (contributes +1).
                    # No need to use a change.
                    current_score += 1
                else: # original_contribution == -1
                    # This is a "bad" move (contributes -1).
                    # If we have changes remaining, we convert it to a "good" move.
                    # Converting a -1 contribution to a +1 contribution yields a net gain of 2.
                    if k_remaining > 0:
                        current_score += 1 # Convert -1 to +1
                        k_remaining -= 1
                    else:
                        # No changes remaining, must take the original -1 hit.
                        current_score -= 1
                
                # Update the maximum score observed for this specific linear combination.
                # This maximum score represents one of the four components (x+y, x-y, -x+y, -x-y)
                # that might ultimately yield the maximum Manhattan distance.
                max_score_for_this_type = max(max_score_for_this_type, current_score)
            
            # The maximum value achieved for this linear combination contributes to the
            # overall maximum Manhattan distance.
            overall_max_dist = max(overall_max_dist, max_score_for_this_type)
        
        return overall_max_dist