# YOUR CODE HERE
import sys
import threading
def main():
    import bisect
    import sys
    sys.setrecursionlimit(1 << 25)
    
    N = int(sys.stdin.readline())
    X = list(map(int, sys.stdin.readline().split()))
    Q = int(sys.stdin.readline())
    tasks = [tuple(map(int, sys.stdin.readline().split())) for _ in range(Q)]
    
    # Initialize positions, note that we need to track current positions of each person
    # Positions are maintained in a sorted list
    positions = X[:]
    person_indices = [i for i in range(N)]  # index of person at positions[i]
    pos_person = dict(zip(positions, person_indices))
    positions_set = set(positions)
    
    from bisect import bisect_left, bisect_right, insort_left

    total_movement = 0

    import bisect

    # Use sorted list to maintain positions
    from bisect import bisect_left, bisect_right

    for task in tasks:
        T_i, G_i = task
        T_i -=1  # 0-based index
        current_pos = positions[T_i]

        if current_pos == G_i:
            continue
        
        d = 1 if G_i > current_pos else -1

        # We will move the person from current_pos to G_i, and move any blocking people along the way

        path_positions = []
        pos = current_pos
        while True:
            pos += d
            path_positions.append(pos)
            if pos == G_i:
                break
        

        # Move the people along the path
        for pos in path_positions:
            if pos in positions_set:
                # Need to move the person at pos
                idx = positions.index(pos)
                # Move that person one step further
                new_pos = pos
                while new_pos in positions_set:
                    new_pos += d
                    total_movement += 1
                positions_set.remove(positions[idx])
                positions[idx] = new_pos
                positions_set.add(new_pos)
                pos_person[new_pos] = pos_person.pop(pos)
            # Now move our person into pos
        positions_set.remove(current_pos)
        positions[T_i] = G_i
        positions_set.add(G_i)
        total_movement += abs(current_pos - G_i)
        
    print(total_movement)
threading.Thread(target=main).start()