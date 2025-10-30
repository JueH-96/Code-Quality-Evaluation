def solve():
    s = input()
    t = input()
    
    def is_airport_code_method1(s, t):
        if len(t) != 3:
            return False
        target_chars = [c.lower() for c in t]
        start_index = 0
        found_indices = []
        for target_char in target_chars:
            found_index = -1
            for i in range(start_index, len(s)):
                if s[i] == target_char:
                    found_index = i
                    break
            if found_index == -1:
                return False
            else:
                found_indices.append(found_index)
                start_index = found_index + 1
        return True
        
    def is_airport_code_method2(s, t):
        if len(t) != 3:
            return False
        if t[2] != 'X':
            return False
        target_chars = [c.lower() for c in t[:2]]
        start_index = 0
        found_indices = []
        for target_char in target_chars:
            found_index = -1
            for i in range(start_index, len(s)):
                if s[i] == target_char:
                    found_index = i
                    break
            if found_index == -1:
                return False
            else:
                found_indices.append(found_index)
                start_index = found_index + 1
        return True
        
    if is_airport_code_method1(s, t) or is_airport_code_method2(s, t):
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    solve()