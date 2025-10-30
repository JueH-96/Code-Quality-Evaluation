import sys
import bisect

def main():
    input = sys.stdin.read
    data = input().split()
    Q = int(data[0])
    index = 1
    plants = []
    current_time = 0
    output = []
    
    for _ in range(Q):
        query_type = data[index]
        if query_type == '1':
            # Query type 1: Plant a new plant with height 0
            plants.append(-current_time)
            index += 1
        elif query_type == '2':
            # Query type 2: Wait for T days
            T = int(data[index+1])
            current_time += T
            index += 2
        elif query_type == '3':
            # Query type 3: Harvest plants with height >= H
            H = int(data[index+1])
            # Calculate the threshold: -current_time + H
            threshold = -current_time + H
            # Find the number of plants with height >= H
            # Since plants are stored as -current_time when planted, their current height is current_time + (-plant_time)
            # So, current_height = current_time + plant_time
            # To find plants with current_height >= H, we need current_time + plant_time >= H
            # Since plant_time is stored as -plant_time, we have current_time - (-plant_time) >= H
            # Which is current_time + plant_time >= H
            # So, plant_time >= H - current_time
            # Since plant_time is stored as -plant_time, we need -plant_time >= H - current_time
            # Which is plant_time <= current_time - H
            # So, we need to find the number of plants with plant_time <= current_time - H
            # Since plants are stored in sorted order, we can use bisect_right
            pos = bisect.bisect_right(plants, current_time - H)
            count = len(plants) - pos
            output.append(str(count))
            # Remove the harvested plants
            plants = plants[:pos]
            index += 2
        else:
            index += 1
    
    print("
".join(output))

if __name__ == "__main__":
    main()