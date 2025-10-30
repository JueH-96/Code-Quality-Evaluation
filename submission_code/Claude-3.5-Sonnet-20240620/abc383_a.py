# YOUR CODE HERE
def solve():
    N = int(input())
    events = []
    for _ in range(N):
        T, V = map(int, input().split())
        events.append((T, V))
    
    water = 0
    prev_time = 0
    
    for time, volume in events:
        # Calculate water loss since last event
        water = max(0, water - (time - prev_time))
        # Add new water
        water += volume
        prev_time = time
    
    print(water)

solve()