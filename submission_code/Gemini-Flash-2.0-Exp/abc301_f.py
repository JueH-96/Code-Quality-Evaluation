def solve():
    s = input()
    n = len(s)
    mod = 998244353
    
    q_count = s.count('?')
    total_strings = pow(52, q_count, mod)
    
    def is_ddos(sub):
        if len(sub) != 4:
            return False
        if not (sub[0].isupper() and sub[1].isupper() and sub[2].islower() and sub[3].isupper()):
            return False
        if sub[0] != sub[1]:
            return False
        return True
    
    def count_ddos_subsequences(string):
        count = 0
        for i in range(len(string) - 3):
            for j in range(i + 1, len(string) - 2):
                for k in range(j + 1, len(string) - 1):
                    for l in range(k + 1, len(string)):
                        sub = string[i] + string[j] + string[k] + string[l]
                        if is_ddos(sub):
                            count += 1
        return count
    
    def generate_strings(index, current_string, strings):
        if index == n:
            strings.append(current_string)
            return
        
        if s[index] == '?':
            for char_code in range(ord('A'), ord('Z') + 1):
                generate_strings(index + 1, current_string + chr(char_code), strings)
            for char_code in range(ord('a'), ord('z') + 1):
                generate_strings(index + 1, current_string + chr(char_code), strings)
        else:
            generate_strings(index + 1, current_string + s[index], strings)

    
    
    def contains_ddos_subsequence(string):
        for i in range(len(string) - 3):
            for j in range(i + 1, len(string) - 2):
                for k in range(j + 1, len(string) - 1):
                    for l in range(k + 1, len(string)):
                        sub = string[i] + string[j] + string[k] + string[l]
                        if is_ddos(sub):
                            return True
        return False
    
    
    
    count_without_ddos = 0
    
    if q_count <= 6:
        strings = []
        generate_strings(0, "", strings)
        
        for string in strings:
            if not contains_ddos_subsequence(string):
                count_without_ddos += 1
        
        print(count_without_ddos % mod)
    else:
        
        dp = {}
        
        def solve_dp(index, a, b, c):
            if index == n:
                return 1
            
            if (index, a, b, c) in dp:
                return dp[(index, a, b, c)]
            
            ans = 0
            
            chars = []
            if s[index] == '?':
                for char_code in range(ord('A'), ord('Z') + 1):
                    chars.append(chr(char_code))
                for char_code in range(ord('a'), ord('z') + 1):
                    chars.append(chr(char_code))
            else:
                chars.append(s[index])
            
            for char in chars:
                new_a = a
                new_b = b
                new_c = c
                
                if a == 0:
                    if 'A' <= char <= 'Z':
                        new_a = 1
                elif a == 1:
                    if 'A' <= char <= 'Z':
                        new_b = 1
                elif b == 1:
                    if 'a' <= char <= 'z':
                        new_c = 1
                elif c == 1:
                    if 'A' <= char <= 'Z':
                        pass
                
                if a == 1 and b == 1 and c == 1 and 'A' <= char <= 'Z' and s[index] != '?':
                    pass
                elif a == 1 and b == 1 and c == 1 and 'A' <= char <= 'Z' and s[index] == '?':
                    pass
                elif a == 1 and b == 1 and 'a' <= char <= 'z' and s[index] != '?':
                    pass
                elif a == 1 and b == 1 and 'a' <= char <= 'z' and s[index] == '?':
                    pass
                elif a == 1 and 'A' <= char <= 'Z' and s[index] != '?':
                    pass
                elif a == 1 and 'A' <= char <= 'Z' and s[index] == '?':
                    pass
                elif 'A' <= char <= 'Z' and s[index] != '?':
                    pass
                elif 'A' <= char <= 'Z' and s[index] == '?':
                    pass
                
                if a == 1 and b == 1 and c == 1 and 'A' <= char <= 'Z':
                    if s[index] != '?':
                        pass
                    else:
                        pass
                elif a == 1 and b == 1 and 'a' <= char <= 'z':
                    if s[index] != '?':
                        pass
                    else:
                        pass
                elif a == 1 and 'A' <= char <= 'Z':
                    if s[index] != '?':
                        pass
                    else:
                        pass
                elif 'A' <= char <= 'Z':
                    if s[index] != '?':
                        pass
                    else:
                        pass
                
                
                if a == 1 and b == 1 and c == 1 and 'A' <= char <= 'Z':
                    if s[index] != '?':
                        if s[index].isupper():
                            pass
                        else:
                            pass
                    else:
                        pass
                elif a == 1 and b == 1 and 'a' <= char <= 'z':
                    if s[index] != '?':
                        if s[index].islower():
                            pass
                        else:
                            pass
                    else:
                        pass
                elif a == 1 and 'A' <= char <= 'Z':
                    if s[index] != '?':
                        if s[index].isupper():
                            pass
                        else:
                            pass
                    else:
                        pass
                elif 'A' <= char <= 'Z':
                    if s[index] != '?':
                        if s[index].isupper():
                            pass
                        else:
                            pass
                    else:
                        pass
                
                if a == 1 and b == 1 and c == 1 and 'A' <= char <= 'Z':
                    if s[index] != '?':
                        if s[index].isupper():
                            if s[index-1].isupper() and s[index-2].isupper() and s[index-3].islower() and s[index-1] == s[index-2]:
                                pass
                            else:
                                ans = (ans + solve_dp(index + 1, new_a, new_b, new_c)) % mod
                        else:
                            ans = (ans + solve_dp(index + 1, new_a, new_b, new_c)) % mod
                    else:
                        ans = (ans + solve_dp(index + 1, new_a, new_b, new_c)) % mod
                elif a == 1 and b == 1 and 'a' <= char <= 'z':
                    if s[index] != '?':
                        if s[index].islower():
                            if s[index-1].isupper() and s[index-2].isupper() and s[index-3].islower() and s[index-1] == s[index-2]:
                                pass
                            else:
                                ans = (ans + solve_dp(index + 1, new_a, new_b, new_c)) % mod
                        else:
                            ans = (ans + solve_dp(index + 1, new_a, new_b, new_c)) % mod
                    else:
                        ans = (ans + solve_dp(index + 1, new_a, new_b, new_c)) % mod
                elif a == 1 and 'A' <= char <= 'Z':
                    if s[index] != '?':
                        if s[index].isupper():
                            if s[index-1].isupper() and s[index-2].isupper() and s[index-3].islower() and s[index-1] == s[index-2]:
                                pass
                            else:
                                ans = (ans + solve_dp(index + 1, new_a, new_b, new_c)) % mod
                        else:
                            ans = (ans + solve_dp(index + 1, new_a, new_b, new_c)) % mod
                    else:
                        ans = (ans + solve_dp(index + 1, new_a, new_b, new_c)) % mod
                elif 'A' <= char <= 'Z':
                    if s[index] != '?':
                        if s[index].isupper():
                            if s[index-1].isupper() and s[index-2].isupper() and s[index-3].islower() and s[index-1] == s[index-2]:
                                pass
                            else:
                                ans = (ans + solve_dp(index + 1, new_a, new_b, new_c)) % mod
                        else:
                            ans = (ans + solve_dp(index + 1, new_a, new_b, new_c)) % mod
                    else:
                        ans = (ans + solve_dp(index + 1, new_a, new_b, new_c)) % mod
                else:
                    ans = (ans + solve_dp(index + 1, new_a, new_b, new_c)) % mod
            
            dp[(index, a, b, c)] = ans
            return ans
        
        print(solve_dp(0, 0, 0, 0))
        
solve()