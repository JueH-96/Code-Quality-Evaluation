def solve():
    n = int(input())
    a_values = list(map(int, input().split()))
    b_values = list(map(int, input().split()))
    
    valid_sequences_count = 0
    
    import itertools
    
    permutations_n = list(itertools.permutations(range(1, n + 1)))
    
    for sigma1_tuple in permutations_n:
        for sigma2_tuple in permutations_n:
            for sigma3_tuple in permutations_n:
                sigma1 = list(sigma1_tuple)
                sigma2 = list(sigma2_tuple)
                sigma3 = list(sigma3_tuple)
                
                valid_sigma = True
                for i in range(n):
                    if sigma1[i] == sigma3[i]:
                        valid_sigma = False
                        break
                if not valid_sigma:
                    continue
                    
                s_sequences = []
                t_sequences = []
                for i in range(n):
                    s_sequences.append(tuple([sigma1[i], sigma2[i], sigma3[i]]))
                    t_sequences.append(tuple([sigma3[i], sigma2[i], sigma1[i]]))
                    
                sequences_set = set(s_sequences + t_sequences)
                if len(sequences_set) != 2 * n:
                    continue
                    
                all_sequences = sorted(list(sequences_set))
                index_map_s = {}
                index_map_t = {}
                for i in range(n):
                    index_map_s[s_sequences[i]] = i + 1
                    index_map_t[t_sequences[i]] = i + 1
                    
                a_result = [0] * n
                b_result = [0] * n
                
                for i in range(n):
                    s_index = all_sequences.index(s_sequences[i]) + 1
                    t_index = all_sequences.index(t_sequences[i]) + 1
                    a_result[i] = s_index
                    b_result[i] = t_index
                    
                match_conditions = True
                for i in range(n):
                    if a_result[i] != a_values[i]:
                        match_conditions = False
                        break
                    if b_values[i] != -1 and b_result[i] != b_values[i]:
                        match_conditions = False
                        break
                if match_conditions:
                    valid_sequences_count += 1
                    
    print(valid_sequences_count % 998244353)

if __name__ == '__main__':
    solve()