from bisect import bisect_right

q = int(input())
wait_time = 0
plant_birth_times = []
output = []

for _ in range(q):
    query = input().split()
    type = int(query[0])

    if type == 1:
        plant_birth_times.append(wait_time)
    elif type == 2:
        t = int(query[1])
        wait_time += t
    elif type == 3:
        h = int(query[1])
        threshold = wait_time - h
        plant_birth_times.sort()
        harvest_index = bisect_right(plant_birth_times, threshold)
        output.append(harvest_index)
        plant_birth_times = plant_birth_times[harvest_index:]

for res in output:
    print(res)