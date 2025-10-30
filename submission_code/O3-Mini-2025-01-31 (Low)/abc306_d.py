def main():
    import sys
    input_data = sys.stdin.buffer.read().split()
    if not input_data:
        return
    it = iter(input_data)
    n = int(next(it))
    
    # dp_healthy: maximum tastiness sum so far when Takahashi currently has a healthy stomach.
    # dp_upset: maximum tastiness sum so far when his stomach is upset.
    # Initially, he is healthy and has eaten nothing.
    dp_healthy = 0
    dp_upset = float("-inf")
    
    for _ in range(n):
        # Read current course information.
        # Course type: 0 => antidotal, 1 => poisonous.
        course_type = int(next(it))
        tastiness = int(next(it))
        
        if course_type == 0:  # antidotal course
            # Option to skip: state remains unchanged.
            cand_healthy = dp_healthy    # healthy state if skipped (stays healthy)
            cand_upset = dp_upset        # upset state if skipped (stays upset)
            # Option to eat:
            #   If current state is healthy: eating an antidote keeps him healthy.
            #   If current state is upset: eating an antidote makes him healthy.
            if dp_healthy != float("-inf"):
                cand_healthy = max(cand_healthy, dp_healthy + tastiness)
            if dp_upset != float("-inf"):
                cand_healthy = max(cand_healthy, dp_upset + tastiness)
                
            new_dp_healthy = cand_healthy
            # There is no way to become upset by eating an antidote.
            new_dp_upset = cand_upset
            
        else:  # poisonous course
            # Option to skip: state remains unchanged.
            cand_healthy = dp_healthy    # remain healthy if already healthy
            cand_upset = dp_upset        # remain upset if already upset
            # Option to eat:
            #   If he is healthy, eating a poisonous course makes him upset.
            #   Eating a poisonous course in upset state would kill him,
            #   so that transition is not allowed.
            if dp_healthy != float("-inf"):
                cand_upset = max(cand_upset, dp_healthy + tastiness)
                
            new_dp_healthy = cand_healthy
            new_dp_upset = cand_upset
            
        dp_healthy, dp_upset = new_dp_healthy, new_dp_upset
        
    # At the end, as long as Takahashi is alive his state can be healthy or upset.
    answer = max(dp_healthy, dp_upset)
    # If no course is eaten, answer should be 0.
    if answer < 0:
        answer = 0
    sys.stdout.write(str(answer))

if __name__ == '__main__':
    main()