def solve():
    n, m, k = map(int, input().split())
    x = list(map(int, input().split()))
    
    is_consecutive_equal = False
    for i in range(m - 1):
        if x[i] == x[i+1]:
            is_consecutive_equal = True
            break
    
    distinct_values_in_x = set(x)
    if is_consecutive_equal and len(distinct_values_in_x) >= 2:
        print(0)
        return
        
    valid_sequences = 0
    
    def is_subsequence(sequence, target_subsequence):
        seq_len = len(sequence)
        sub_len = len(target_subsequence)
        seq_idx = 0
        sub_idx = 0
        while seq_idx < seq_len and sub_idx < sub_len:
            if sequence[seq_idx] == target_subsequence[sub_idx]:
                sub_idx += 1
            seq_idx += 1
        return sub_idx == sub_len
        
    all_m_sequences = []
    
    def generate_sequences(current_sequence):
        if len(current_sequence) == m:
            all_m_sequences.append(tuple(current_sequence))
            return
        for i in range(1, k + 1):
            current_sequence.append(i)
            generate_sequences(current_sequence)
            current_sequence.pop()
            
    generate_sequences([])
    x_tuple = tuple(x)
    other_m_sequences = [seq for seq in all_m_sequences if seq != x_tuple]
    
    def check_conditions(a_sequence):
        if is_subsequence(a_sequence, x):
            return False
        for y in other_m_sequences:
            if not is_subsequence(a_sequence, y):
                return False
        return True
        
    def get_sequences(current_sequence):
        nonlocal valid_sequences
        if len(current_sequence) == n:
            if check_conditions(tuple(current_sequence)):
                valid_sequences += 1
            return
        for i in range(1, k + 1):
            current_sequence.append(i)
            get_sequences(current_sequence)
            current_sequence.pop()
            
    get_sequences([])
    print(valid_sequences % 998244353)

if __name__ == '__main__':
    solve()