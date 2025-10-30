def solve():
    s = input()
    n = len(s)
    mod = 998244353
    
    q_count = s.count('?')
    
    total_strings = pow(52, q_count, mod)
    
    
    def is_ddos_type(sub):
      if len(sub) != 4:
        return False
      if not ('A' <= sub[0] <= 'Z' and 'A' <= sub[1] <= 'Z' and 'a' <= sub[2] <= 'z' and 'A' <= sub[3] <= 'Z'):
        return False
      if sub[0] != sub[1]:
        return False
      return True

    def count_non_ddos_strings(current_s, index):
        if index == n:
            
            
            
            
            for i in range(len(current_s) - 3):
              if is_ddos_type(current_s[i:i+4]):
                return 0
            return 1
        
        if s[index] != '?':
            return count_non_ddos_strings(current_s + s[index], index + 1)
        else:
            count = 0
            for char_code in range(ord('A'), ord('Z') + 1):
                count = (count + count_non_ddos_strings(current_s + chr(char_code), index + 1)) % mod
            for char_code in range(ord('a'), ord('z') + 1):
                count = (count + count_non_ddos_strings(current_s + chr(char_code), index + 1)) % mod
            return count

    
    result = count_non_ddos_strings("",0)
    print(result)

solve()