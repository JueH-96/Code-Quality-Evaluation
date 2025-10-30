import sys
import heapq

def read_input():
    N = int(sys.stdin.readline().strip())
    D = list(map(int, sys.stdin.readline().strip().split()))
    sensors = [list(map(int, sys.stdin.readline().strip().split())) for _ in range(2)]
    return N, D, sensors

def solve(N, D, sensors):
    # Sort the sensors by their length in ascending order
    sensors.sort()
    # Use a heap to store the lengths of the sensors that have not been used yet
    heap = []
    for i in range(2):
        for _ in range(sensors[i][2]):
            heapq.heappush(heap, sensors[i][0])
    # Try to use the sensors to cover the sections
    total_cost = 0
    for d in D:
        while heap and heap[0] < d:
            heapq.heappop(heap)
        if not heap:
            return -1
        total_cost += heapq.heappop(heap)
    return total_cost

def main():
    N, D, sensors = read_input()
    result = solve(N, D, sensors)
    print(result)

if __name__ == "__main__":
    main()