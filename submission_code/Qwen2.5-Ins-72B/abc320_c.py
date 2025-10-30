# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    M = int(data[0])
    S1 = data[1]
    S2 = data[2]
    S3 = data[3]
    
    # Create a dictionary to store the positions of each character in each reel
    positions = {}
    for i, char in enumerate(S1):
        if char not in positions:
            positions[char] = []
        positions[char].append(i)
    
    for i, char in enumerate(S2):
        if char not in positions:
            positions[char] = []
        positions[char].append(i)
    
    for i, char in enumerate(S3):
        if char not in positions:
            positions[char] = []
        positions[char].append(i)
    
    # Find the minimum time to align the same character on all reels
    min_time = float('inf')
    for char, pos in positions.items():
        if len(pos) == 3:
            # Calculate the time to align the same character on all reels
            time1 = pos[0]
            time2 = pos[1]
            time3 = pos[2]
            max_time = max(time1, time2, time3)
            min_time = min(min_time, max_time)
    
    if min_time == float('inf'):
        print(-1)
    else:
        print(min_time)

if __name__ == "__main__":
    main()