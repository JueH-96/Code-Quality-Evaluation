import sys
import heapq

def can_monitor_all_sections(N, D, L, C, K):
    # Priority queue to store the cost and type of sensors
    pq = []

    # Add all possible sensors to the priority queue
    for j in range(2):
        for _ in range(K[j]):
            heapq.heappush(pq, (C[j], L[j]))

    total_cost = 0
    for length in D:
        current_length = 0
        temp_cost = 0

        # Use a temporary list to store used sensors for the current section
        temp_sensors = []

        while current_length < length:
            if not pq:
                return False
            cost, sensor_length = heapq.heappop(pq)
            temp_sensors.append((cost, sensor_length))
            temp_cost += cost
            current_length += sensor_length

        # Add the cost of used sensors to the total cost
        total_cost += temp_cost

        # Put back the used sensors into the priority queue
        for sensor in temp_sensors:
            heapq.heappush(pq, sensor)

    return total_cost

def main():
    input = sys.stdin.read
    data = input().split()

    index = 0
    N = int(data[index])
    index += 1
    D = list(map(int, data[index:index + N]))
    index += N
    L = [int(data[index + 1]), int(data[index + 4])]
    C = [int(data[index + 2]), int(data[index + 5])]
    K = [int(data[index + 3]), int(data[index + 6])]

    result = can_monitor_all_sections(N, D, L, C, K)
    if result is False:
        print(-1)
    else:
        print(result)

if __name__ == "__main__":
    main()