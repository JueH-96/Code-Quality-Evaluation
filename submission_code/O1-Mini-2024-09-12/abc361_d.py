# YOUR CODE HERE
from collections import deque
import sys

def encode_state(S, N):
    state = 0
    for i in range(N + 2):
        if i < N:
            if S[i] == 'W':
                code = 1
            elif S[i] == 'B':
                code = 2
            else:
                code = 0
        else:
            code = 0
        state |= (code << (2 * i))
    return state

def get_moves(state, N):
    sources = []
    for x in range(N +1):
        b0 = (state >> (2 * x)) & 3
        b1 = (state >> (2 * (x +1))) & 3
        if b0 !=0 and b1 !=0:
            sources.append(x)
    targets = []
    for k in range(N +1):
        t0 = (state >> (2 * k)) & 3
        t1 = (state >> (2 * (k +1))) & 3
        if t0 ==0 and t1 ==0:
            targets.append(k)
    for x in sources:
        for k in targets:
            if abs(x - k) >1:
                stone_x = (state >> (2 * x)) & 3
                stone_x1 = (state >> (2 * (x +1))) & 3
                new_state = (state & ~(3 << (2 * x)) & ~(3 << (2 * (x +1)))) | (stone_x << (2 * k)) | (stone_x1 << (2 * (k +1)))
                yield new_state

def bfs_bidirectional(start, target, N):
    if start == target:
        return 0
    queue1 = deque()
    queue2 = deque()
    visited1 = {}
    visited2 = {}
    queue1.append(start)
    visited1[start] = 0
    queue2.append(target)
    visited2[target] = 0
    while queue1 and queue2:
        # Expand the BFS with fewer nodes
        if len(queue1) <= len(queue2):
            current_queue = queue1
            current_visited = visited1
            other_visited = visited2
        else:
            current_queue = queue2
            current_visited = visited2
            other_visited = visited1
        for _ in range(len(current_queue)):
            current = current_queue.popleft()
            current_distance = current_visited[current]
            for next_state in get_moves(current, N):
                if next_state in current_visited:
                    continue
                if next_state in other_visited:
                    return current_distance + 1 + other_visited[next_state]
                current_visited[next_state] = current_distance +1
                current_queue.append(next_state)
    return -1

def main():
    input = sys.stdin.read().split()
    N = int(input[0])
    S = input[1]
    T = input[2]
    start = encode_state(S, N)
    target = encode_state(T, N)
    result = bfs_bidirectional(start, target, N)
    print(result)

if __name__ == "__main__":
    main()