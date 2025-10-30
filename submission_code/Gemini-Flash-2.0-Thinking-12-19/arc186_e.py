def is_subsequence(s, t):
    s_ptr = 0
    t_ptr = 0
    while s_ptr < len(s) and t_ptr < len(t):
        if s[s_ptr] == t[t_ptr]:
            s_ptr += 1
        t_ptr += 1
    return s_ptr == len(s)

def solve():
    n, m, k = map(int, input().split())
    x_seq = list(map(int, input().split()))
    
    count = 0
    mod = 998244353
    
    sequences_to_check = []
    def generate_sequences(current_sequence):
        if len(current_sequence) == m:
            seq = tuple(current_sequence)
            if seq != tuple(x_seq):
                sequences_to_check.append(seq)
            return
        for i in range(1, k + 1):
            current_sequence.append(i)
            generate_sequences(current_sequence)
            current_sequence.pop()
            
    generate_sequences([])
    
    valid_sequences = []
    
    def get_sequences_length_n(current_sequence):
        nonlocal count, valid_sequences
        if len(current_sequence) == n:
            a_seq = tuple(current_sequence)
            is_x_subsequence = is_subsequence(x_seq, a_seq)
            if is_x_subsequence:
                return
                
            all_others_subsequences = True
            for y_seq in sequences_to_check:
                if not is_subsequence(y_seq, a_seq):
                    all_others_subsequences = False
                    break
                    
            if all_others_subsequences:
                count += 1
                valid_sequences.append(a_seq)
                
            return
            
        for i in range(1, k + 1):
            current_sequence.append(i)
            get_sequences_length_n(current_sequence)
            current_sequence.pop()
            
    get_sequences_length_n([])
    
    print(count % mod)

if __name__ == '__main__':
    solve()