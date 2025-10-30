def solve():
    n, m = map(int, input().split())
    strings = [input() for _ in range(n)]
    
    def are_adjacent(s1, s2):
        diff_count = 0
        for i in range(m):
            if s1[i] != s2[i]:
                diff_count += 1
        return diff_count == 1
        
    adjacency_list = {s: [] for s in strings}
    for i in range(n):
        for j in range(i + 1, n):
            if are_adjacent(strings[i], strings[j]):
                adjacency_list[strings[i]].append(strings[j])
                adjacency_list[strings[j]].append(strings[i])
                
    def find_hamiltonian_path(current_string, visited_strings):
        if len(visited_strings) == n:
            return True
        for neighbor in adjacency_list[current_string]:
            if neighbor not in visited_strings:
                visited_strings.add(neighbor)
                if find_hamiltonian_path(neighbor, visited_strings):
                    return True
                visited_strings.remove(neighbor) # Backtrack
        return False
        
    for start_string in strings:
        if find_hamiltonian_path(start_string, {start_string}):
            print("Yes")
            return
            
    print("No")

if __name__ == '__main__':
    solve()