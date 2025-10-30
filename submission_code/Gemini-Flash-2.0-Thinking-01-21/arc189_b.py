import collections

def solve():
    n = int(input())
    x_initial = list(map(int, input().split()))
    current_coordinates = sorted(x_initial)
    initial_state = tuple(current_coordinates)
    min_sum_coordinates = sum(current_coordinates)
    visited_states = {initial_state}
    queue = collections.deque([current_coordinates])
    
    while queue:
        current_coords = queue.popleft()
        current_sum = sum(current_coords)
        min_sum_coordinates = min(min_sum_coordinates, current_sum)
        
        for i in range(n - 3):
            x_i = current_coords[i]
            x_i_plus_1 = current_coords[i+1]
            x_i_plus_2 = current_coords[i+2]
            x_i_plus_3 = current_coords[i+3]
            
            next_coords_list = list(current_coords)
            next_coords_list[i+1] = x_i + x_i_plus_3 - x_i_plus_1
            next_coords_list[i+2] = x_i + x_i_plus_3 - x_i_plus_2
            
            next_coords_sorted = sorted(next_coords_list)
            next_state = tuple(next_coords_sorted)
            
            if next_state not in visited_states:
                visited_states.add(next_state)
                queue.append(next_coords_sorted)
                
    print(min_sum_coordinates)

if __name__ == '__main__':
    solve()