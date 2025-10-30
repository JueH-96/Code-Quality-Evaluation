# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    
    # Prepare a list to store the changes in pill count
    changes = {}
    
    index = 2
    for _ in range(N):
        a_i = int(data[index])
        b_i = int(data[index + 1])
        index += 2
        
        # Apply the difference array technique
        if 1 not in changes:
            changes[1] = 0
        changes[1] += b_i
        
        end_day = a_i + 1
        if end_day not in changes:
            changes[end_day] = 0
        changes[end_day] -= b_i
    
    # Calculate the prefix sum to find the total pills taken each day
    current_pills = 0
    day = 1
    
    # We need to check from day 1 onwards
    while True:
        if day in changes:
            current_pills += changes[day]
        
        if current_pills <= K:
            print(day)
            return
        
        day += 1