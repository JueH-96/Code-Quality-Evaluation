def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    courses = [(int(data[i*2+1]), int(data[i*2+2])) for i in range(N)]
    
    # dp_healthy and dp_upset to track the maximum tastiness in both states
    dp_healthy = 0  # Maximum tastiness sum when stomach is healthy
    dp_upset = float('-inf')  # Maximum tastiness sum when stomach is upset (initially impossible)
    
    for x, y in courses:
        if x == 0:  # antidotal course
            new_dp_healthy = max(dp_healthy, dp_healthy + y)
            new_dp_upset = max(dp_upset, dp_upset + y)
        else:  # poisonous course
            new_dp_healthy = max(dp_healthy, dp_upset + y)
            new_dp_upset = dp_healthy + y
        
        dp_healthy = new_dp_healthy
        dp_upset = new_dp_upset
    
    # The answer is the maximum tastiness sum when Takahashi is not dead, i.e., healthy or upset but not dead
    print(max(dp_healthy, 0))

if __name__ == "__main__":
    main()