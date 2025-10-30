def diff_by_one(s1, s2):
    if len(s1) != len(s2):
        return False
    diff_count = 0
    for i in range(len(s1)):
        if s1[i] != s2[i]:
            diff_count += 1
    return diff_count == 1

def solve():
    n, m = map(int, input().split())
    strings = [input() for _ in range(n)]
    string_set = set(strings)
    
    def find_path(current_path):
        if len(current_path) == n:
            return True
        last_string = current_path[-1] if current_path else None
        
        for next_string in list(string_set):
            if next_string in current_path:
                continue
            if last_string is None or diff_by_one(last_string, next_string):
                current_path.append(next_string)
                string_set.remove(next_string)
                if find_path(current_path):
                    return True
                string_set.add(next_string)
                current_path.pop()
        return False
        
    for start_string in list(strings):
        initial_path = [start_string]
        string_set_initial = set(strings)
        string_set_initial.remove(start_string)
        string_set_global = string_set_initial.copy()
        if find_path(initial_path):
            print("Yes")
            return
            
    print("No")

if __name__ == '__main__':
    solve()