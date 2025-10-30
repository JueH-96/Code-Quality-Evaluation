import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    bases = []
    
    index = 1
    for _ in range(N):
        W = int(data[index])
        X = int(data[index + 1])
        bases.append((W, X))
        index += 2
    
    # We need to find the maximum overlap of working hours in UTC
    # Convert each base's working hours to UTC time
    # Base i: works from (9-X_i)%24 to (18-X_i)%24 in UTC
    
    # We will use a sweep line algorithm with event points
    # We will consider each hour as a potential starting point for the meeting (0 to 23)
    
    # We need to adjust the times to handle wrap around the 24 hour clock
    # For each base, calculate the UTC times they can attend the meeting
    # and then adjust for 24 hour wrap around if needed
    
    # We will use a list to count the number of employees that can attend starting at each hour
    employee_count = [0] * 24
    
    for W, X in bases:
        start_time = (9 - X) % 24
        end_time = (18 - X) % 24
        
        if start_time <= end_time:
            # No wrap around
            for hour in range(start_time, end_time + 1):
                employee_count[hour] += W
        else:
            # Wrap around the clock
            for hour in range(start_time, 24):
                employee_count[hour] += W
            for hour in range(0, end_time + 1):
                employee_count[hour] += W
    
    # The maximum number of employees that can attend the meeting
    max_employees = max(employee_count)
    
    print(max_employees)

if __name__ == "__main__":
    main()