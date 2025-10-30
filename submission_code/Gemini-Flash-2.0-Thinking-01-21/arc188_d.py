import collections

def solve():
    n = int(input())
    a_input = list(map(int, input().split()))
    b_input = list(map(int, input().split()))
    
    count = 0
    
    def get_permutations(elements):
        if not elements:
            yield []
        else:
            for i in range(len(elements)):
                rest_elements = elements[:i] + elements[i+1:]
                for perm in get_permutations(rest_elements):
                    yield [elements[i]] + perm
                    
    numbers = list(range(1, n + 1))
    
    sigma1_permutations = list(get_permutations(numbers))
    sigma2_permutations = list(get_permutations(numbers))
    sigma3_permutations = list(get_permutations(numbers))
    
    for sigma1 in sigma1_permutations:
        for sigma3 in sigma3_permutations:
            valid_sigma1_sigma3 = True
            for i in range(n):
                if sigma1[i] == sigma3[i]:
                    valid_sigma1_sigma3 = False
                    break
            if not valid_sigma1_sigma3:
                continue
                
            s_sequences = []
            t_sequences = []
            for i in range(n):
                s_sequences.append((sigma1[i], sigma2_permutations[0][i], sigma3[i]))
                t_sequences.append((sigma3[i], sigma2_permutations[0][i], sigma1[i]))
                
            all_sequences = []
            for i in range(n):
                all_sequences.append(s_sequences[i])
                all_sequences.append(t_sequences[i])
                
            if len(set(all_sequences)) < 2 * n:
                continue
                
            sigma1_inv = {}
            for i in range(n):
                sigma1_inv[sigma1[i]] = i + 1
            sigma3_inv = {}
            for i in range(n):
                sigma3_inv[sigma3[i]] = i + 1
                
            valid_choices_count = 0
            
            for choice_int in range(1 << n):
                valid_choice = True
                relations = []
                current_a = [0] * n
                current_b = [0] * n
                
                for v_val in range(1, n + 1):
                    i_v = sigma1_inv[v_val] - 1
                    j_v = sigma3_inv[v_val] - 1
                    
                    relation_choice = (choice_int >> (v_val - 1)) & 1
                    if relation_choice == 0: # s < t, i.e., sigma2(i_v) < sigma2(j_v)
                        expected_a_rank = 2 * v_val - 1
                        expected_b_rank = 2 * v_val
                        if a_input[i_v] != expected_a_rank:
                            valid_choice = False
                            break
                        if b_input[j_v] != -1 and b_input[j_v] != expected_b_rank:
                            valid_choice = False
                            break
                        relations.append((i_v, j_v))
                        current_a[i_v] = expected_a_rank
                        current_b[j_v] = expected_b_rank
                    else: # t < s, i.e., sigma2(i_v) > sigma2(j_v)
                        expected_a_rank = 2 * v_val
                        expected_b_rank = 2 * v_val - 1
                        if a_input[i_v] != expected_a_rank:
                            valid_choice = False
                            break
                        if b_input[j_v] != -1 and b_input[j_v] != expected_b_rank:
                            valid_choice = False
                            break
                        relations.append((j_v, i_v))
                        current_a[i_v] = expected_a_rank
                        current_b[j_v] = expected_b_rank
                    if not valid_choice:
                        break
                if not valid_choice:
                    continue
                    
                dag = True
                adj = collections.defaultdict(list)
                for u, v in relations:
                    adj[u].append(v)
                    
                def has_cycle(graph):
                    visited = [0] * n # 0: unvisited, 1: visiting, 2: visited
                    for i in range(n):
                        if visited[i] == 0:
                            if is_cyclic_util(i, visited, graph):
                                return True
                    return False
                    
                def is_cyclic_util(u, visited, graph):
                    visited[u] = 1 # Mark as visiting
                    for v in graph[u]:
                        if visited[v] == 1:
                            return True # Cycle detected
                        if visited[v] == 0:
                            if is_cyclic_util(v, visited, graph):
                                return True
                    visited[u] = 2 # Mark as visited
                    return False
                    
                if has_cycle(adj):
                    dag = False
                    
                if dag:
                    valid_choices_count += 1

            count = (count + valid_choices_count) % 998244353

    print(count)

if __name__ == '__main__':
    solve()