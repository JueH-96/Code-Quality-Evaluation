import sys
import threading

def main():
    import sys
    import heapq

    data = sys.stdin
    line = data.readline()
    if not line:
        return
    n = int(line)
    tasks = []
    # Read and build (release, deadline) pairs
    # A print at time t is allowed if T_i <= t <= T_i + D_i.
    # After printing at t, next print can be at t+1 or later.
    for _ in range(n):
        t_i, d_i = map(int, data.readline().split())
        # release time r = t_i, latest print time = t_i + d_i
        tasks.append((t_i, t_i + d_i))
    # Sort by release time
    tasks.sort(key=lambda x: x[0])

    ans = 0
    t = 0      # current time
    idx = 0    # next task to consider by release time
    heap = []  # min-heap of deadlines of "available" tasks

    while idx < n or heap:
        # Push all tasks whose release time <= current time
        while idx < n and tasks[idx][0] <= t:
            # push its deadline
            heapq.heappush(heap, tasks[idx][1])
            idx += 1
        # Drop any tasks whose deadline < current time (can't schedule them)
        while heap and heap[0] < t:
            heapq.heappop(heap)
        if heap:
            # Schedule the available task with the earliest deadline
            heapq.heappop(heap)
            ans += 1
            # Reserve [t, t+1) for cooldown, next print at t+1 or later
            t += 1
        else:
            # No task available right now: jump to next release time
            if idx < n:
                t = tasks[idx][0]
            else:
                # No more tasks at all
                break

    # Output the maximum number of prints
    print(ans)

if __name__ == "__main__":
    main()