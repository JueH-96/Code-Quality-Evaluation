def main():
    import sys
    input_data = sys.stdin.buffer.read().split()
    if not input_data:
        return
    it = iter(input_data)
    n = int(next(it))
    
    # We need a sentinel for negative infinity.
    # The worst-case sum magnitude is roughly 10^9 * 3*10^5 = 3e14,
    # so we choose a sentinel well below that.
    NEG_INF = -10**20
    
    # dp0: max tastiness sum when stomach is healthy.
    # dp1: max tastiness sum when stomach is upset.
    dp0, dp1 = 0, NEG_INF  # Initially, his stomach is healthy with sum 0.
    
    for _ in range(n):
        x = int(next(it))
        y = int(next(it))
        
        # Prepare new states for the next course.
        new0, new1 = NEG_INF, NEG_INF
        
        # Option 1: Skip the course.
        # Skipping does not change the stomach state.
        new0 = max(new0, dp0)
        new1 = max(new1, dp1)
        
        if x == 0:
            # Antidotal course:
            # Eating it makes the stomach healthy regardless of the previous state.
            if dp0 != NEG_INF:
                new0 = max(new0, dp0 + y)
            if dp1 != NEG_INF:
                new0 = max(new0, dp1 + y)
        else:
            # Poisonous course:
            # Can only be eaten safely when the stomach is healthy.
            # If eaten from a healthy state, his stomach becomes upset.
            if dp0 != NEG_INF:
                new1 = max(new1, dp0 + y)
            # From an upset state, eating a poisonous course is fatal.
            # Hence, we do not transition from dp1 for x == 1.
        
        dp0, dp1 = new0, new1

    # At the end, Takahashi survives if he is either in a healthy or upset state.
    # We also allow the possibility of eating nothing, which gives 0 tastiness.
    answer = max(dp0, dp1, 0)
    sys.stdout.write(str(answer))

if __name__ == '__main__':
    main()