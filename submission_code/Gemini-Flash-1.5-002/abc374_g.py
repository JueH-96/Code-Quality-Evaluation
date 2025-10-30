def solve():
    n = int(input())
    s = [input() for _ in range(n)]
    
    adj = {}
    for si in s:
        adj.setdefault(si[0], []).append(si[1])
        
    
    def check(ng_list):
        covered = set()
        for ng_str in ng_list:
            for i in range(len(ng_str) - 1):
                covered.add(ng_str[i:i+2])
        return covered == set(s)

    def is_valid(ng_list):
        for ng_str in ng_list:
            for i in range(len(ng_str) - 1):
                if ng_str[i:i+2] not in s:
                    return False
        return True

    
    def find_min_strings(current_strings, covered_strings):
        if check(current_strings):
            return len(current_strings)
        
        min_strings = float('inf')
        
        
        for i in range(1 << n):
            next_strings = list(current_strings)
            new_string = ""
            
            for j in range(n):
                if (i >> j) & 1:
                    new_string += s[j]
            
            if new_string:
                next_strings.append(new_string)
                if is_valid(next_strings):
                    min_strings = min(min_strings, find_min_strings(next_strings, covered_strings | set(s)))

        return min_strings

    print(find_min_strings([], set()))

solve()