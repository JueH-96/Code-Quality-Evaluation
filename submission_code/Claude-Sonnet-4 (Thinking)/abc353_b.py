n, k = map(int, input().split())
groups = list(map(int, input().split()))

empty_seats = k
attraction_starts = 0
i = 0

while i < n:
    front_group = groups[i]
    
    if empty_seats < front_group:
        # Start the attraction
        attraction_starts += 1
        empty_seats = k
    else:
        # Guide the front group
        empty_seats -= front_group
        i += 1

# Start the attraction one more time to end
attraction_starts += 1

print(attraction_starts)