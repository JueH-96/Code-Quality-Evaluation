def solve():
    N, K = map(int, input().split())
    S = list(input().strip())
    
    MOD = 998244353
    
    # Find positions of ?s
    question_positions = [i for i in range(N) if S[i] == '?']
    num_questions = len(question_positions)
    
    def is_good(a_count, b_count, c_count):
        return a_count % 2 == b_count % 2 == c_count % 2
    
    result = 0
    
    # Generate all possible ways to fill ?s
    def generate_fillings(index):
        if index == num_questions:
            # All ?s filled, now count good substrings
            good_count = 0
            for i in range(N):
                a_count = b_count = c_count = 0
                for j in range(i, N):
                    if S[j] == 'A':
                        a_count += 1
                    elif S[j] == 'B':
                        b_count += 1
                    elif S[j] == 'C':
                        c_count += 1
                    
                    if is_good(a_count, b_count, c_count):
                        good_count += 1
            
            if good_count >= K:
                nonlocal result
                result = (result + 1) % MOD
            return
        
        # Fill the index-th ? with A, B, or C
        pos = question_positions[index]
        for char in ['A', 'B', 'C']:
            S[pos] = char
            generate_fillings(index + 1)
            S[pos] = '?'  # Restore
    
    generate_fillings(0)
    print(result)

solve()