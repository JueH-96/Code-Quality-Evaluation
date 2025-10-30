# YOUR CODE HERE
import sys
import bisect

def main():
    import sys
    import threading

    def solve():
        import sys

        sys.setrecursionlimit(1 << 25)
        N = int(sys.stdin.readline())
        X = list(map(int, sys.stdin.readline().split()))
        Q = int(sys.stdin.readline())
        tasks = [tuple(map(int, sys.stdin.readline().split())) for _ in range(Q)]

        # Initialize
        pos_map = {}  # position -> person_id
        person_pos = [0] * (N +1)  # person_id from1 to N
        for i in range(N):
            pos_map[X[i]] = i +1
            person_pos[i +1] = X[i]

        sorted_positions = sorted(X)

        total_movement =0

        for task in tasks:
            T_i, G_i = task
            current_pos = person_pos[T_i]
            if G_i > current_pos:
                # Move East
                l = bisect.bisect_right(sorted_positions, current_pos)
                r = bisect.bisect_right(sorted_positions, G_i)
                overlapping_pos = sorted_positions[l:r]
                y = G_i +1
                for pos in overlapping_pos:
                    y = max(y, pos +1, y)
                    person = pos_map[pos]
                    movement = y - pos
                    total_movement += movement
                    # Assign person to y
                    pos_map[y] = person
                    person_pos[person] = y
                    bisect.insort(sorted_positions, y)
                    # Remove old pos
                    del pos_map[pos]
                    idx = bisect.bisect_left(sorted_positions, pos)
                    if idx < len(sorted_positions) and sorted_positions[idx] == pos:
                        sorted_positions.pop(idx)
                    y +=1
                # Assign T_i to G_i
                movement = abs(G_i - current_pos)
                total_movement += movement
                pos_map[G_i] = T_i
                bisect.insort(sorted_positions, G_i)
                # Remove old pos
                del pos_map[current_pos]
                idx = bisect.bisect_left(sorted_positions, current_pos)
                if idx < len(sorted_positions) and sorted_positions[idx] == current_pos:
                    sorted_positions.pop(idx)
                person_pos[T_i] = G_i
            elif G_i < current_pos:
                # Move West
                l = bisect.bisect_left(sorted_positions, G_i)
                r = bisect.bisect_left(sorted_positions, current_pos)
                overlapping_pos = sorted_positions[l:r]
                y = G_i -1
                for pos in reversed(overlapping_pos):
                    y = min(y, pos -1, y)
                    person = pos_map[pos]
                    movement = pos - y
                    total_movement += movement
                    # Assign person to y
                    pos_map[y] = person
                    person_pos[person] = y
                    bisect.insort(sorted_positions, y)
                    # Remove old pos
                    del pos_map[pos]
                    idx = bisect.bisect_left(sorted_positions, pos)
                    if idx < len(sorted_positions) and sorted_positions[idx] == pos:
                        sorted_positions.pop(idx)
                    y -=1
                # Assign T_i to G_i
                movement = abs(G_i - current_pos)
                total_movement += movement
                pos_map[G_i] = T_i
                bisect.insort(sorted_positions, G_i)
                # Remove old pos
                del pos_map[current_pos]
                idx = bisect.bisect_left(sorted_positions, current_pos)
                if idx < len(sorted_positions) and sorted_positions[idx] == current_pos:
                    sorted_positions.pop(idx)
                person_pos[T_i] = G_i
            else:
                # G_i == current_pos
                pass
        print(total_movement)

    threading.Thread(target=solve).start()

if __name__ == "__main__":
    main()