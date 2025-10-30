# YOUR CODE HERE
def solve():
    s = input()
    n = len(s)
    mod = 998244353
    
    q_count = s.count('?')
    total_strings = pow(52, q_count, mod)
    
    def count_ddos_subsequences(string):
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        if 'A' <= string[i] <= 'Z' and 'A' <= string[j] <= 'Z' and 'a' <= string[k] <= 'z' and 'A' <= string[l] <= 'Z' and string[i] == string[j]:
                            count += 1
        return count

    
    def generate_strings(s, index, current_string, ddos_strings):
        if index == len(s):
            if count_ddos_subsequences(current_string) == 0:
                ddos_strings.append(current_string)
            return
        
        if s[index] == '?':
            for char_code in range(ord('A'), ord('Z') + 1):
                generate_strings(s, index + 1, current_string + chr(char_code), ddos_strings)
            for char_code in range(ord('a'), ord('z') + 1):
                generate_strings(s, index + 1, current_string + chr(char_code), ddos_strings)
        else:
            generate_strings(s, index + 1, current_string + s[index], ddos_strings)

    
    if q_count <= 18:
        ddos_strings = []
        generate_strings(s, 0, "", ddos_strings)
        print(len(ddos_strings) % mod)
    else:
        dp = {}

        def count_valid(i, has_d1, has_d2, has_o):
            if i == n:
                return 1
            
            if (i, has_d1, has_d2, has_o) in dp:
                return dp[(i, has_d1, has_d2, has_o)]
            
            ans = 0
            
            if s[i] == '?':
                for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz":
                    new_d1 = has_d1
                    new_d2 = has_d2
                    new_o = has_o

                    if not has_d1 and 'A' <= char <= 'Z':
                        new_d1 = True
                    elif has_d1 and not has_d2 and char == s[i-1] if s[i-1] != '?' else 'A' <= char <= 'Z':
                        new_d2 = True
                    elif has_d1 and has_d2 and not has_o and 'a' <= char <= 'z':
                        new_o = True
                    elif has_d1 and has_d2 and has_o and 'A' <= char <= 'Z':
                        continue
                    
                    ans = (ans + count_valid(i+1, new_d1, new_d2, new_o)) % mod
            else:
                new_d1 = has_d1
                new_d2 = has_d2
                new_o = has_o

                if not has_d1 and 'A' <= s[i] <= 'Z':
                    new_d1 = True
                elif has_d1 and not has_d2 and s[i] == s[i-1] if s[i-1] != '?' else 'A' <= s[i] <= 'Z':
                    new_d2 = True
                elif has_d1 and has_d2 and not has_o and 'a' <= s[i] <= 'z':
                    new_o = True
                elif has_d1 and has_d2 and has_o and 'A' <= s[i] <= 'Z':
                    ans = 0
                    dp[(i, has_d1, has_d2, has_o)] = ans
                    return ans
                
                ans = (ans + count_valid(i+1, new_d1, new_d2, new_o)) % mod
            
            dp[(i, has_d1, has_d2, has_o)] = ans
            return ans

        print(count_valid(0, False, False, False))

solve()