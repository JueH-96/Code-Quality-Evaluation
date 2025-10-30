def solve():
    n, m = map(int, input().split())
    s = input()
    t = input()
    
    initial_x = ['#'] * n
    queue = ["".join(initial_x)]
    visited = {queue[0]}
    
    while queue:
        current_x_str = queue.pop(0)
        if current_x_str == s:
            print("Yes")
            return
            
        for i in range(n - m + 1):
            next_x_list = list(current_x_str)
            next_x_list[i:i+m] = list(t)
            next_x_str = "".join(next_x_list)
            if next_x_str not in visited:
                visited.add(next_x_str)
                queue.append(next_x_str)
                
    print("No")

if __name__ == '__main__':
    solve()