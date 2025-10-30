# YOUR CODE HERE
Q = int(input())
plants = []

for _ in range(Q):
    query = list(map(int, input().split()))
    
    if query[0] == 1:
        # Add a new plant with height 0
        plants.append(0)
    elif query[0] == 2:
        # All plants grow by T
        T = query[1]
        for i in range(len(plants)):
            plants[i] += T
    else:  # query[0] == 3
        # Harvest all plants with height >= H
        H = query[1]
        count = sum(1 for height in plants if height >= H)
        plants = [height for height in plants if height < H]
        print(count)