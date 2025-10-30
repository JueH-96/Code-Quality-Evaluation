import math

def gcd(a, b):
    if b == 0:
        return a
    return gcd(b, a % b)

def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    good_sequences = []
    
    def find_good_sequences(current_sequence):
        if len(current_sequence) == n:
            # Check if the sequence is good
            valid = True
            for i in range(n - 1):
                p = current_sequence[i]
                q = current_sequence[i+1]
                g = gcd(p, q)
                p_reduced = p // g
                q_reduced = q // g
                if p_reduced * q_reduced != a[i]:
                    valid = False
                    break
            
            if valid:
                # Check if gcd(S_1, S_2, ..., S_N) = 1
                seq_gcd = current_sequence[0]
                for i in range(1, n):
                    seq_gcd = gcd(seq_gcd, current_sequence[i])
                
                if seq_gcd == 1:
                    good_sequences.append(current_sequence)
            return
        
        if not current_sequence:
            for i in range(1, 101):
                find_good_sequences([i])
        else:
            last_element = current_sequence[-1]
            for i in range(1, 101):
                find_good_sequences(current_sequence + [i])

    find_good_sequences([])
    
    total_score = 0
    for seq in good_sequences:
        score = 1
        for x in seq:
            score = (score * x) % 998244353
        total_score = (total_score + score) % 998244353
    
    print(total_score)

solve()