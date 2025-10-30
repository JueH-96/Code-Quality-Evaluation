import collections

def solve():
    n, m = map(int, input().split())
    s = input()
    t = input()
    
    initial_x = ['#'] * n
    target_s_list = list(s)
    
    queue = collections.deque()
    queue.append(tuple(initial_x))
    visited_states = set()
    visited_states.add(tuple(initial_x))
    
    while queue:
        current_x_tuple = queue.popleft()
        current_x_list = list(current_x_tuple)
        
        if current_x_list == target_s_list:
            print("Yes")
            return
            
        for i in range(n - m + 1):
            next_x_list = list(current_x_list)
            for j in range(m):
                next_x_list[i+j] = t[j]
            next_x_tuple = tuple(next_x_list)
            if next_x_tuple not in visited_states:
                visited_states.add(next_x_tuple)
                queue.append(next_x_tuple)
                
    print("No")

if __name__ == '__main__':
    solve()