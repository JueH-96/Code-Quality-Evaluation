def main():
    import sys
    
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    
    INF_NEG = float('-inf')
    
    healthy = 0   # dp state for "healthy stomach"
    upset = INF_NEG  # dp state for "upset stomach" (start impossible)
    
    idx = 1
    for _ in range(N):
        x = int(input_data[idx]); y = int(input_data[idx+1])
        idx += 2
        
        # Skip transitions (carry over previous dp state)
        new_healthy = healthy
        new_upset = upset
        
        # Eat transitions:
        if x == 0:
            # Antidotal course
            if healthy != INF_NEG:
                new_healthy = max(new_healthy, healthy + y)
            if upset != INF_NEG:
                new_healthy = max(new_healthy, upset + y)
        else:
            # Poisonous course
            if healthy != INF_NEG:
                new_upset = max(new_upset, healthy + y)
        
        healthy, upset = new_healthy, new_upset
    
    # The maximum sum while remaining alive (healthy or upset) at the end
    print(max(healthy, upset))

# Do not forget to call main()
if __name__ == "__main__":
    main()