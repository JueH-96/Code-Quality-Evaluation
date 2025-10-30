allowed_set = set("atcoder")

def main():
    s = input().strip()
    t = input().strip()
    
    cntS = [0] * 128
    cntT = [0] * 128
    
    for char in s:
        cntS[ord(char)] += 1
    for char in t:
        cntT[ord(char)] += 1
        
    for c_code in range(ord('a'), ord('z')+1):
        char = chr(c_code)
        if char not in allowed_set:
            if cntS[c_code] != cntT[c_code]:
                print("No")
                return
                
    a_count = cntS[ord('@')]
    b_count = cntT[ord('@')]
    
    atcoder_chars = "atcoder"
    N = 0
    P = 0
    for char in atcoder_chars:
        c_code = ord(char)
        if cntT[c_code] > cntS[c_code]:
            N += cntT[c_code] - cntS[c_code]
        if cntS[c_code] > cntT[c_code]:
            P += cntS[c_code] - cntT[c_code]
            
    if a_count >= N and b_count >= P:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()