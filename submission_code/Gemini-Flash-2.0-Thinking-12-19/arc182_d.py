import collections

def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    
    start_sequence = tuple(a)
    target_sequence = tuple(b)
    
    if start_sequence == target_sequence:
        print(0)
        return
        
    queue = collections.deque([start_sequence])
    distance = {start_sequence: 0}
    visited_sequences = {start_sequence}
    
    while queue:
        current_sequence = queue.popleft()
        if current_sequence == target_sequence:
            print(distance[current_sequence])
            return
            
        for i in range(n):
            # Increment operation
            next_val_inc = (current_sequence[i] + 1) % m
            is_valid_inc = True
            if i > 0 and next_val_inc == current_sequence[i-1]:
                is_valid_inc = False
            if i < n - 1 and next_val_inc == current_sequence[i+1]:
                is_valid_inc = False
                
            if is_valid_inc:
                next_sequence_list_inc = list(current_sequence)
                next_sequence_list_inc[i] = next_val_inc
                next_sequence_inc = tuple(next_sequence_list_inc)
                if next_sequence_inc not in visited_sequences:
                    visited_sequences.add(next_sequence_inc)
                    distance[next_sequence_inc] = distance[current_sequence] + 1
                    queue.append(next_sequence_inc)
                    
            # Decrement operation
            next_val_dec = (current_sequence[i] - 1) % m
            is_valid_dec = True
            if i > 0 and next_val_dec == current_sequence[i-1]:
                is_valid_dec = False
            if i < n - 1 and next_val_dec == current_sequence[i+1]:
                is_valid_dec = False
                
            if is_valid_dec:
                next_sequence_list_dec = list(current_sequence)
                next_sequence_list_dec[i] = next_val_dec
                next_sequence_dec = tuple(next_sequence_list_dec)
                if next_sequence_dec not in visited_sequences:
                    visited_sequences.add(next_sequence_dec)
                    distance[next_sequence_dec] = distance[current_sequence] + 1
                    queue.append(next_sequence_dec)
                    
    print(-1)

if __name__ == '__main__':
    solve()