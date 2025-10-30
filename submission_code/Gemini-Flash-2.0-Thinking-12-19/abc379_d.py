import bisect

plant_base_heights = []
current_base_height = 0
output_lines = []

q = int(input())
for _ in range(q):
    query = list(map(int, input().split()))
    query_type = query[0]
    if query_type == 1:
        bisect.insort(plant_base_heights, current_base_height)
    elif query_type == 2:
        t = query[1]
        current_base_height += t
    elif query_type == 3:
        h = query[1]
        threshold_base_height = current_base_height - h
        harvest_index = bisect.bisect_right(plant_base_heights, threshold_base_height)
        harvest_count = harvest_index
        output_lines.append(harvest_count)
        plant_base_heights = plant_base_heights[harvest_index:]
        
for count in output_lines:
    print(count)