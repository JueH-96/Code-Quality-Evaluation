import sys

def main():
    k = int(sys.stdin.readline())
    s = sys.stdin.readline().strip()
    t = sys.stdin.readline().strip()
    
    if s == t:
        print("Yes")
        return
    
    len_s = len(s)
    len_t = len(t)
    
    if abs(len_s - len_t) > 1:
        print("No")
        return
    
    if len_s == len_t:
        diff = 0
        for a, b in zip(s, t):
            if a != b:
                diff += 1
                if diff > 1:
                    break
        print("Yes" if diff <= 1 else "No")
    else:
        if len_s > len_t:
            long_str, short_str = s, t
        else:
            long_str, short_str = t, s
        
        i = j = 0
        count = 0
        len_long = len(long_str)
        len_short = len(short_str)
        
        while i < len_long and j < len_short:
            if long_str[i] == short_str[j]:
                i += 1
                j += 1
            else:
                if count == 1:
                    print("No")
                    return
                count += 1
                i += 1
        
        if j == len_short and count <= 1:
            print("Yes")
        else:
            print("No")

if __name__ == "__main__":
    main()