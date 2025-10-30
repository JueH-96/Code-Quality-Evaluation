import math

def solve():
    n, s, t = map(int, input().split())
    segments = []
    for _ in range(n):
        a, b, c, d = map(int, input().split())
        segments.append(((a, b), (c, d)))
    
    endpoints = [(0, 0)] # index 0 is starting point (0, 0)
    for i in range(n):
        endpoints.append(segments[i][0]) # index 2i-1, endpoint U_i
        endpoints.append(segments[i][1]) # index 2i, endpoint V_i
        
    memo = {}
    
    def get_distance(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        
    def get_min_time(mask, last_endpoint_index):
        if mask == (1 << n) - 1:
            return 0
        if (mask, last_endpoint_index) in memo:
            return memo[(mask, last_endpoint_index)]
        
        min_time = float('inf')
        
        for i in range(n):
            if not (mask & (1 << i)):
                u_index = 2 * (i + 1) - 1
                v_index = 2 * (i + 1)
                u_point = endpoints[u_index]
                v_point = endpoints[v_index]
                current_pos = endpoints[last_endpoint_index] if last_endpoint_index != 0 else (0, 0)
                
                # Print from U_i to V_i
                move_time = get_distance(current_pos, u_point) / s
                print_time = get_distance(u_point, v_point) / t
                next_mask = mask | (1 << i)
                remaining_time = get_min_time(next_mask, v_index)
                total_time = move_time + print_time + remaining_time
                min_time = min(min_time, total_time)
                
                # Print from V_i to U_i
                move_time = get_distance(current_pos, v_point) / s
                print_time = get_distance(v_point, u_point) / t
                remaining_time = get_min_time(next_mask, u_index)
                total_time = move_time + print_time + remaining_time
                min_time = min(min_time, total_time)
                
        memo[(mask, last_endpoint_index)] = min_time
        return min_time
        
    result = get_min_time(0, 0)
    print(f"{result:.12f}")

if __name__ == '__main__':
    solve()