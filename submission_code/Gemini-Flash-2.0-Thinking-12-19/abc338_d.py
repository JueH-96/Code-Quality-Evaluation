def solve():
    n, m = map(int, input().split())
    x = list(map(int, input().split()))
    min_tour_length = float('inf')
    
    for bridge_to_remove_index in range(1, n + 1):
        current_tour_length = 0
        for i in range(m - 1):
            u = x[i]
            v = x[i+1]
            if bridge_to_remove_index <= n - 1:
                u_index = ((u - bridge_to_remove_index - 1) % n) + 1
                v_index = ((v - bridge_to_remove_index - 1) % n) + 1
                distance = abs(u_index - v_index)
            else: # bridge_to_remove_index == n
                distance = abs(u - v)
            current_tour_length += distance
        min_tour_length = min(min_tour_length, current_tour_length)
        
    print(min_tour_length)

if __name__ == '__main__':
    solve()