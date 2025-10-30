def solve():
    n = int(input())
    a = list(map(int, input().split()))
    results = []
    for k in range(1, n + 1):
        initial_slimes = list(a)
        takahashi_initial_index = k - 1
        
        memo = {}
        
        def get_max_size(current_slimes, takahashi_index):
            state = (tuple(current_slimes), takahashi_index)
            if state in memo:
                return memo[state]
            
            current_size = current_slimes[takahashi_index]
            possible_directions = []
            if takahashi_index > 0 and current_slimes[takahashi_index - 1] < current_size:
                possible_directions.append('left')
            if takahashi_index < len(current_slimes) - 1 and current_slimes[takahashi_index + 1] < current_size:
                possible_directions.append('right')
                
            if not possible_directions:
                return current_size
                
            max_final_size = current_size
            for direction in possible_directions:
                next_slimes = list(current_slimes)
                next_takahashi_index = takahashi_index
                if direction == 'left':
                    absorbed_size = next_slimes[takahashi_index - 1]
                    next_size = current_size + absorbed_size
                    del next_slimes[takahashi_index - 1]
                    next_slimes[takahashi_index - 1] = next_size
                    next_takahashi_index = takahashi_index - 1
                elif direction == 'right':
                    absorbed_size = next_slimes[takahashi_index + 1]
                    next_size = current_size + absorbed_size
                    del next_slimes[takahashi_index + 1]
                    next_slimes[takahashi_index] = next_size
                    
                final_size = get_max_size(next_slimes, next_takahashi_index)
                max_final_size = max(max_final_size, final_size)
                
            memo[state] = max_final_size
            return max_final_size
            
        max_size_k = get_max_size(initial_slimes, takahashi_initial_index)
        results.append(max_size_k)
        
    print(*(results))

if __name__ == '__main__':
    solve()