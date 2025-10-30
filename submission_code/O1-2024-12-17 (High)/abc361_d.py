from collections import deque

def main():
    N = int(input().strip())
    S = input().strip()
    T = input().strip()
    
    # Quick impossibility checks
    # 1) Same number of black stones must exist in S and T.
    if S.count('B') != T.count('B'):
        print(-1)
        return
    
    # 2) If S already equals T, zero moves are needed.
    if S == T:
        print(0)
        return
    
    start = S + '..'
    goal = T + '..'
    
    # A helper function to generate all neighboring states (one move away).
    def neighbors(state):
        # state is a string of length N+2
        i = state.find('..')  # find the contiguous empty pair
        arr = list(state)
        out = []
        # For each adjacent pair of stones, move it to the empty spot
        for x in range(N+1):
            if arr[x] != '.' and arr[x+1] != '.':
                new_arr = arr[:]
                new_arr[i], new_arr[i+1] = arr[x], arr[x+1]
                new_arr[x], new_arr[x+1] = '.', '.'
                out.append(''.join(new_arr))
        return out

    # Standard bidirectional BFS setup
    visited_start = {start: 0}
    visited_goal = {goal: 0}
    queue_start = deque([start])
    queue_goal = deque([goal])
    
    # Expand one step in BFS from one side
    def expand(q, visited_this, visited_other):
        size = len(q)
        for _ in range(size):
            cur = q.popleft()
            dist_cur = visited_this[cur]
            for nxt in neighbors(cur):
                if nxt not in visited_this:
                    visited_this[nxt] = dist_cur + 1
                    # If this state is already visited from the other side, we have a meet
                    if nxt in visited_other:
                        return visited_this[nxt] + visited_other[nxt]
                    q.append(nxt)
        return None
    
    # Bidirectional BFS loop
    while queue_start and queue_goal:
        # Always expand the side with fewer states in its frontier
        if len(queue_start) <= len(queue_goal):
            res = expand(queue_start, visited_start, visited_goal)
        else:
            res = expand(queue_goal, visited_goal, visited_start)
        
        if res is not None:
            print(res)
            return
    
    # If we exhaust both frontiers without meeting, it's impossible
    print(-1)

# Do not forget to call main()
if __name__ == "__main__":
    main()