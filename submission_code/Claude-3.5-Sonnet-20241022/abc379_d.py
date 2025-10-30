# Read number of queries
Q = int(input())

# Initialize list to store plants (their heights)
plants = []

# Process each query
for _ in range(Q):
    query = input().split()
    query_type = int(query[0])
    
    if query_type == 1:
        # Plant a new flower with height 0
        plants.append(0)
    
    elif query_type == 2:
        # Increase height of all plants by T
        T = int(query[1])
        plants = [height + T for height in plants]
    
    else:  # query_type == 3
        # Harvest all plants with height >= H
        H = int(query[1])
        # Count plants to harvest
        harvest_count = sum(1 for height in plants if height >= H)
        # Remove harvested plants
        plants = [height for height in plants if height < H]
        # Print number of harvested plants
        print(harvest_count)