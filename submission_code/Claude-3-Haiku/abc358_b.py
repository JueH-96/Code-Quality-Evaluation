# Read the input
N, A = map(int, input().split())
T = list(map(int, input().split()))

# Initialize the current time and the end time of the last person in line
current_time = 0
end_times = []

# Process each person
for i in range(N):
    # If there is no one in line, the person starts immediately
    if not end_times:
        end_time = T[i] + A
    # Otherwise, the person joins the end of the line
    else:
        end_time = max(current_time, end_times[-1]) + A
    end_times.append(end_time)
    current_time = end_time
    
    # Print the end time for the current person
    print(end_time)