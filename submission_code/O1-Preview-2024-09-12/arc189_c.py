# YOUR CODE HERE
import sys
import threading
def main():
    import sys
    import sys
    sys.setrecursionlimit(1 << 25)
    N, X = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    P = list(map(int, sys.stdin.readline().split()))
    Q = list(map(int, sys.stdin.readline().split()))
    X -= 1  # zero-based index
    from collections import defaultdict

    # For red balls
    visited_red = [False]*N
    cycle_id_red = [0]*N  # To identify cycles
    pos_in_cycle_red = [0]*N  # Position in cycle
    cycles_red = []
    cycle_num = 0

    for i in range(N):
        if not visited_red[i]:
            curr = i
            cycle = []
            idx_in_cycle = {}
            idx = 0
            while not visited_red[curr]:
                visited_red[curr] = True
                cycle.append(curr)
                idx_in_cycle[curr] = idx
                pos_in_cycle_red[curr] = idx
                idx +=1
                curr = P[curr]-1  # zero-based index
            cycles_red.append(cycle)
            for node in cycle:
                cycle_id_red[node] = cycle_num
            cycle_num +=1
    # Similarly for blue balls
    visited_blue = [False]*N
    cycle_id_blue = [0]*N  # To identify cycles
    pos_in_cycle_blue = [0]*N  # Position in cycle
    cycles_blue = []
    cycle_num = 0

    for i in range(N):
        if not visited_blue[i]:
            curr = i
            cycle = []
            idx_in_cycle = {}
            idx = 0
            while not visited_blue[curr]:
                visited_blue[curr] = True
                cycle.append(curr)
                idx_in_cycle[curr] = idx
                pos_in_cycle_blue[curr] = idx
                idx +=1
                curr = Q[curr]-1  # zero-based index
            cycles_blue.append(cycle)
            for node in cycle:
                cycle_id_blue[node] = cycle_num
            cycle_num +=1

    total_operations = 0

    # For red balls
    for i in range(N):
        if A[i]:
            if cycle_id_red[i]!=cycle_id_red[X]:
                print(-1)
                return
            else:
                length = len(cycles_red[cycle_id_red[i]])
                pos_i = pos_in_cycle_red[i]
                pos_X = pos_in_cycle_red[X]
                dist = (pos_i - pos_X)%length
                total_operations += dist
    # For blue balls
    for i in range(N):
        if B[i]:
            if cycle_id_blue[i]!=cycle_id_blue[X]:
                print(-1)
                return
            else:
                length = len(cycles_blue[cycle_id_blue[i]])
                pos_i = pos_in_cycle_blue[i]
                pos_X = pos_in_cycle_blue[X]
                dist = (pos_i - pos_X)%length
                total_operations += dist
    print(total_operations)
threading.Thread(target=main).start()