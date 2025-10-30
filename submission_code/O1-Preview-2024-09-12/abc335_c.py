# YOUR CODE HERE

import sys
import threading

def main():
    import sys
    N, Q = map(int, sys.stdin.readline().split())
    queries = [sys.stdin.readline().strip() for _ in range(Q)]
    positions_head = [(1, 0)]
    move_dirs = {'R': (1, 0), 'L': (-1, 0), 'U': (0, 1), 'D': (0, -1)}
    output = []
    t = 0  # time step index for positions_head
    for query in queries:
        q = query.split()
        if q[0] == '1':
            C = q[1]
            dx, dy = move_dirs[C]
            x_prev, y_prev = positions_head[-1]
            x_new = x_prev + dx
            y_new = y_prev + dy
            positions_head.append((x_new, y_new))
            t +=1
        elif q[0] == '2':
            p = int(q[1])
            if t >= p -1:
                idx = t - (p -1)
                x, y = positions_head[idx]
            else:
                x = 1 + (p - ( t + 1 ))
                y = 0
            output.append(f'{x} {y}')
    print('
'.join(output))


if __name__ == "__main__":
    threading.Thread(target=main).start()