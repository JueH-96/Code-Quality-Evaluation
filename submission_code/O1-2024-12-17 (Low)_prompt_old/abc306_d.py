def solve():
    import sys
    input = sys.stdin.readline
    
    N = int(input())
    courses = [tuple(map(int, input().split())) for _ in range(N)]
    
    # dp_healthy: the maximum sum of tastiness if Takahashi ends up with a healthy stomach
    # dp_upset:   the maximum sum of tastiness if Takahashi ends up with an upset stomach
    #
    # We start with a healthy stomach and haven't eaten anything yet. So:
    dp_healthy = 0
    dp_upset = float('-inf')
    
    for x, y in courses:
        # We will compute the next states after considering the current course.
        # next_healthy and next_upset are the new dp values after possibly eating or skipping the course.
        next_healthy = float('-inf')
        next_upset = float('-inf')
        
        # Case 1: Skip the course
        #   Stomach state does not change
        next_healthy = max(next_healthy, dp_healthy)
        next_upset = max(next_upset, dp_upset)
        
        # Case 2: Eat the course
        #   We transition from the old state (dp_healthy or dp_upset) to the new state depending on (x,y).
        
        # If we were healthy before eating:
        if dp_healthy != float('-inf'):
            if x == 0:
                # Eat antidotal (healthy -> healthy)
                next_healthy = max(next_healthy, dp_healthy + y)
            else:
                # Eat poisonous (healthy -> upset)
                next_upset = max(next_upset, dp_healthy + y)
        
        # If we were upset before eating:
        if dp_upset != float('-inf'):
            if x == 0:
                # Eat antidotal (upset -> healthy)
                next_healthy = max(next_healthy, dp_upset + y)
            else:
                # Eat poisonous (upset -> dead) -> invalid, do nothing.
                pass
        
        # Prepare for the next iteration
        dp_healthy, dp_upset = next_healthy, next_upset
    
    # We want Takahashi to survive with the maximum sum of tastiness eaten.
    # Surviving means being either healthy or upset (not dead).
    # Also, it's possible that skipping everything (sum = 0) is optimal.
    print(max(dp_healthy, dp_upset, 0))

# Let's call solve() to complete the requirement.
def main():
    solve()

if __name__ == "__main__":
    main()