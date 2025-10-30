def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    current_sum = 0
    min_prefix_sum = 0
    
    for i in range(n):
        current_sum += a[i]
        min_prefix_sum = min(min_prefix_sum, current_sum)
        
    initial_passengers = max(0, -min_prefix_sum)
    current_passengers = initial_passengers + current_sum
    
    print(current_passengers)

if __name__ == '__main__':
    solve()