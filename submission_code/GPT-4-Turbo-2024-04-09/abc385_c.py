import sys
from collections import defaultdict

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    heights = list(map(int, data[1:]))
    
    if N == 1:
        print(1)
        return
    
    # Dictionary to store heights and their indices
    height_indices = defaultdict(list)
    for index, height in enumerate(heights):
        height_indices[height].append(index)
    
    max_buildings = 1
    
    # For each unique height, try to find the maximum number of buildings that can be chosen
    for indices in height_indices.values():
        if len(indices) == 1:
            max_buildings = max(max_buildings, 1)
            continue
        
        # Try every possible interval based on the first index
        num_indices = len(indices)
        for start_index in range(num_indices):
            for interval_index in range(start_index + 1, num_indices):
                interval = indices[interval_index] - indices[start_index]
                count = 2
                current_position = indices[interval_index] + interval
                
                for further_index in range(interval_index + 1, num_indices):
                    if indices[further_index] == current_position:
                        count += 1
                        current_position += interval
                
                max_buildings = max(max_buildings, count)
    
    print(max_buildings)

if __name__ == "__main__":
    main()