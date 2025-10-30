def solve():
    a, b, c, d = map(int, input().split())
    total_black_area = 0
    for n in range(a, c):
        for m_even in range(b, d, 2):
            cell_x_start = n
            cell_x_end = n + 1
            cell_y_start = m_even
            cell_y_end = m_even + 2
            
            intersection_x_start = max(cell_x_start, a)
            intersection_x_end = min(cell_x_end, c)
            intersection_y_start = max(cell_y_start, b)
            intersection_y_end = min(cell_y_end, d)
            
            intersection_width = max(0, intersection_x_end - intersection_x_start)
            intersection_height = max(0, intersection_y_end - intersection_y_start)
            intersection_area = intersection_width * intersection_height
            
            if intersection_area > 0:
                cell_n = n
                cell_m = m_even // 2
                n_plus_2m_even = (cell_n + 2 * cell_m) % 2 == 0
                n_plus_m_even = (cell_n + cell_m) % 2 == 0
                
                black_area_in_cell = 0
                if n_plus_2m_even:
                    black_area_in_cell = 1
                else:
                    if n_plus_m_even:
                        black_area_in_cell = 1.5
                    else:
                        black_area_in_cell = 0.5
                        
                black_area_fraction = black_area_in_cell / 2.0
                black_area_in_intersection = intersection_area * black_area_fraction
                total_black_area += black_area_in_intersection
                
    print(int(round(total_black_area * 2)))

if __name__ == '__main__':
    solve()