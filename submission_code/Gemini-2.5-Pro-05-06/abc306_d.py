import sys

def main():
    N = int(sys.stdin.readline())
    
    # dp_H: Maximum tastiness Takahashi can achieve ending in a Healthy state.
    # dp_U: Maximum tastiness Takahashi can achieve ending in an Upset state.
    
    # Initially, Takahashi is healthy with 0 tastiness (having eaten nothing).
    dp_H = 0
    # It's impossible to be initially upset without eating anything.
    # float('-inf') represents an unreachable state or an infinitely bad score.
    dp_U = float('-inf')

    for _ in range(N):
        X, Y = map(int, sys.stdin.readline().split())
        
        # Store values from the previous state (after course i-1)
        # These will be used to calculate values for the current state (after course i)
        prev_dp_H = dp_H
        prev_dp_U = dp_U
        
        # Initialize possible values for current state's dp_H and dp_U.
        # These will be updated based on choices (skip or eat).
        # Start with -inf, assuming no valid path yet. If a path existed, it's overwritten.
        current_dp_H = float('-inf')
        current_dp_U = float('-inf')

        # Option 1: Skip the current course
        # If Takahashi skips, his state and tastiness carry over from the previous step.
        # If he was healthy and skips, he remains healthy.
        # (prev_dp_H can't be -inf because it starts at 0 and is always max'd with something)
        current_dp_H = max(current_dp_H, prev_dp_H)
        
        # If he was upset and skips, he remains upset.
        # This is only possible if prev_dp_U was a valid state (not -inf).
        if prev_dp_U != float('-inf'):
            current_dp_U = max(current_dp_U, prev_dp_U)
            
        # Option 2: Eat the current course
        if X == 0: # Antidotal course (type 0)
            # If he was healthy and eats an antidotal course, he remains healthy.
            if prev_dp_H != float('-inf'): # This check is strictly true since prev_dp_H starts at 0
                current_dp_H = max(current_dp_H, prev_dp_H + Y)
            
            # If he had an upset stomach and eats an antidotal course, he becomes healthy.
            if prev_dp_U != float('-inf'):
                current_dp_H = max(current_dp_H, prev_dp_U + Y)
            
            # Eating an antidotal course cannot result in an upset stomach.
            # So, current_dp_U is only affected by the "skip" option if this course is antidotal.

        else: # Poisonous course (type 1)
            # If he was healthy and eats a poisonous course, he gets an upset stomach.
            if prev_dp_H != float('-inf'): # This check is strictly true
                current_dp_U = max(current_dp_U, prev_dp_H + Y)
            
            # If he had an upset stomach and eats a poisonous course, he DIES.
            # This path is forbidden, so it does not update any state.
            
            # Eating a poisonous course cannot result in a healthy stomach.
            # So, current_dp_H is only affected by the "skip" option if this course is poisonous.
            
        # Update dp_H and dp_U to the new maximums for the current state.
        dp_H = current_dp_H
        dp_U = current_dp_U
        
    # After processing all courses, the answer is the maximum tastiness achieved,
    # regardless of whether Takahashi ends up healthy or with an upset stomach.
    # Both dp_H and dp_U represent states where he is alive.
    # max() will correctly handle if one of them is float('-inf').
    # Since dp_H starts at 0 and skipping is always an option that maintains health,
    # dp_H will not be -inf (unless it was initialized to -inf, which it wasn't).
    # The result will be at least 0 if all eating options lead to negative tastiness.
    result = max(dp_H, dp_U)
    
    # The problem asks for an integer output.
    print(int(result))

if __name__ == '__main__':
    main()