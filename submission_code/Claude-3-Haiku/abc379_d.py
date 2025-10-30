import sys

# Read the number of queries
Q = int(input())

# Initialize the number of flower pots and plants
num_flower_pots = 10 ** 100
num_plants = 0
plant_heights = []

# Process the queries
for _ in range(Q):
    query = input().split()
    query_type = int(query[0])

    if query_type == 1:
        # Prepare one empty flower pot and put a plant in it
        num_flower_pots -= 1
        num_plants += 1
        plant_heights.append(0)

    elif query_type == 2:
        # Wait for T days, increasing the height of every existing plant by T
        T = int(query[1])
        for i in range(num_plants):
            plant_heights[i] += T

    elif query_type == 3:
        # Harvest all plants with a height of at least H
        H = int(query[1])
        harvested_plants = 0
        for i in range(num_plants - 1, -1, -1):
            if plant_heights[i] >= H:
                harvested_plants += 1
                num_plants -= 1
                plant_heights.pop(i)
            else:
                break
        print(harvested_plants)