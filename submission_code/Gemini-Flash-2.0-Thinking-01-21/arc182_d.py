from collections import deque

def is_good_sequence(seq):
    n = len(seq)
    if n <= 1:
        return True
    for i in range(n - 1):
        if seq[i] == seq[i+1]:
            return False
    return True

def solve():
    n, m = map(int, input().split())
    a_list = list(map(int, input().split()))
    b_list = list(map(int, input().split()))
    a_tuple = tuple(a_list)
    b_tuple = tuple(b_list)
    
    if not is_good_sequence(a_tuple) or not is_good_sequence(b_tuple):
        raise ValueError("Input sequences are not good")
        
    if a_tuple == b_tuple:
        print(0)
        return
        
    queue = deque([(a_tuple, 0)])
    visited_sequences = {a_tuple}
    
    while queue:
        current_sequence, operations_count = queue.popleft()
        
        if current_sequence == b_tuple:
            print(operations_count)
            return
            
        for i in range(n):
            # Increment
            next_val_inc = (current_sequence[i] + 1) % m
            temp_seq_inc = list(current_sequence)
            temp_seq_inc[i] = next_val_inc
            next_seq_inc = tuple(temp_seq_inc)
            is_valid_inc = True
            if n > 1:
                if i > 0 and next_seq_inc[i] == next_seq_inc[i-1]:
                    is_valid_inc = False
                if i < n - 1 and next_seq_inc[i] == next_seq_inc[i+1]:
                    is_valid_inc = False
            if is_valid_inc:
                if next_seq_inc not in visited_sequences:
                    visited_sequences.add(next_seq_inc)
                    queue.append((next_seq_inc, operations_count + 1))
                    
            # Decrement
            next_val_dec = (current_sequence[i] - 1) % m
            temp_seq_dec = list(current_sequence)
            temp_seq_dec[i] = next_val_dec
            next_seq_dec = tuple(temp_seq_dec)
            is_valid_dec = True
            if n > 1:
                if i > 0 and next_seq_dec[i] == next_seq_dec[i-1]:
                    is_valid_dec = False
                if i < n - 1 and next_seq_dec[i] == next_seq_dec[i+1]:
                    is_valid_dec = False
            if is_valid_dec:
                if next_seq_dec not in visited_sequences:
                    visited_sequences.add(next_seq_dec)
                    queue.append((next_seq_dec, operations_count + 1))
                    
    print("-1")

if __name__ == '__main__':
    solve()