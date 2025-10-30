# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    M = int(data[0])
    S1 = data[1]
    S2 = data[2]
    S3 = data[3]
    
    # Initialize the minimum time to a large number
    min_time = float('inf')
    
    # Check for each digit from '0' to '9'
    for digit in range(10):
        digit = str(digit)
        # Find the earliest time each reel can display the digit
        times = []
        for S in [S1, S2, S3]:
            # Find all positions where the digit appears
            positions = [i for i in range(M) if S[i] == digit]
            if not positions:
                break
            # Calculate the times when the digit can be displayed
            times.append([i + M * k for k in range(3) for i in positions])
        
        if len(times) < 3:
            continue
        
        # Find the minimum time where all reels can display the digit
        for t1 in times[0]:
            for t2 in times[1]:
                for t3 in times[2]:
                    max_time = max(t1, t2, t3)
                    min_time = min(min_time, max_time)
    
    # If min_time is still infinity, it means it's impossible
    if min_time == float('inf'):
        print(-1)
    else:
        print(min_time)

if __name__ == "__main__":
    main()