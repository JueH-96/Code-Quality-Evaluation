import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        print(0)
        return
    
    it = iter(data)
    N = int(next(it))
    X = int(next(it))
    X0 = X - 1
    
    A = [int(next(it)) for _ in range(N)]
    B = [int(next(it)) for _ in range(N)]
    P = [int(next(it)) for _ in range(N)]
    Q = [int(next(it)) for _ in range(N)]
    
    P0 = [x - 1 for x in P]
    Q0 = [x - 1 for x in Q]
    
    visited_red = [False] * N
    red_cycle_set = set()
    cur = X0
    while not visited_red[cur]:
        visited_red[cur] = True
        red_cycle_set.add(cur)
        cur = P0[cur]
    
    visited_blue = [False] * N
    blue_cycle_set = set()
    cur = X0
    while not visited_blue[cur]:
        visited_blue[cur] = True
        blue_cycle_set.add(cur)
        cur = Q0[cur]
    
    for i in range(N):
        if A[i] == 1:
            if i not in red_cycle_set:
                print(-1)
                return
                
    for i in range(N):
        if B[i] == 1:
            if i not in blue_cycle_set:
                print(-1)
                return
                
    linear_red = []
    cur = P0[X0]
    while cur != X0:
        linear_red.append(cur)
        cur = P0[cur]
    
    linear_blue = []
    cur = Q0[X0]
    while cur != X0:
        linear_blue.append(cur)
        cur = Q0[cur]
    
    node_to_index_red = {}
    for idx, node in enumerate(linear_red):
        node_to_index_red[node] = idx
        
    node_to_index_blue = {}
    for idx, node in enumerate(linear_blue):
        node_to_index_blue[node] = idx
        
    min_red = 10**9
    for i in range(N):
        if A[i] == 1 and i != X0 and i in red_cycle_set:
            if i in node_to_index_red:
                idx = node_to_index_red[i]
                if idx < min_red:
                    min_red = idx
                    
    min_blue = 10**9
    for i in range(N):
        if B[i] == 1 and i != X0 and i in blue_cycle_set:
            if i in node_to_index_blue:
                idx = node_to_index_blue[i]
                if idx < min_blue:
                    min_blue = idx
                    
    red_segment = set()
    if min_red != 10**9:
        red_segment = set(linear_red[min_red:])
        
    blue_segment = set()
    if min_blue != 10**9:
        blue_segment = set(linear_blue[min_blue:])
        
    total_set = red_segment | blue_segment
    print(len(total_set))

if __name__ == "__main__":
    main()