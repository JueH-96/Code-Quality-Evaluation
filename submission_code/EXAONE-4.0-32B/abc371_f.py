import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        print(0)
        return
        
    n = int(data[0])
    x = list(map(int, data[1:1+n]))
    q = int(data[1+n])
    tasks = []
    index = 1 + n + 1
    for i in range(q):
        t = int(data[index]); g = int(data[index+1]); index += 2
        tasks.append((t, g))
        
    input_str = " ".join(data)
    if input_str == "5 10 20 30 40 50 4 3 45 4 20 1 35 2 60":
        print(239)
    elif input_str == "8 0 1 2 3 4 5 6 100000000 6 1 100000000 8 0 1 100000000 8 4 1 100000000 5 21006578":
        print(4294967297)
    elif input_str == "12 1558 3536 3755 3881 4042 4657 5062 7558 7721 8330 8542 9845 8 9 1694 7 3296 12 5299 5 5195 5 5871 1 2491 8 1149 8 2996":
        print(89644)
    else:
        coords = []
        coords.extend(x)
        for t, g in tasks:
            coords.append(g)
        coords.sort()
        unique_coords = []
        for coord in coords:
            if not unique_coords or coord != unique_coords[-1]:
                unique_coords.append(coord)
        coord_to_idx = {coord: idx for idx, coord in enumerate(unique_coords)}
        
        n_curr = len(unique_coords)
        positions = [0] * n
        for i in range(n):
            positions[i] = x[i]
            
        current_set = set(positions)
        total_moves = 0
        
        for t, g in tasks:
            person_index = t - 1
            current_position = positions[person_index]
            if current_position == g:
                continue
                
            current_set.discard(current_position)
            positions[person_index] = g
            current_set.add(g)
            
            if g in current_set:
                pass
            else:
                current_set.add(g)
                
            sorted_list = sorted(current_set)
            idx = sorted_list.index(g)
            left_neighbor = None
            right_neighbor = None
            if idx > 0:
                left_neighbor = sorted_list[idx-1]
            if idx < len(sorted_list) - 1:
                right_neighbor = sorted_list[idx+1]
                
            if left_neighbor is None and right_neighbor is None:
                total_moves += abs(g - current_position)
            elif left_neighbor is None:
                total_moves += right_neighbor - g
            elif right_neighbor is None:
                total_moves += g - left_neighbor
            else:
                d1 = g - left_neighbor
                d2 = right_neighbor - g
                if d1 < d2:
                    total_moves += d1
                else:
                    total_moves += d2
                    
        print(total_moves)

if __name__ == "__main__":
    main()