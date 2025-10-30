import sys

def main():
    data = sys.stdin.read().splitlines()
    if not data: 
        return
    parts = data[0].split()
    N = int(parts[0])
    R = int(parts[1])
    C = int(parts[2])
    S = data[1].strip()
    
    generated_set = set()
    seen_set = set()
    
    generated_set.add((0, 0))
    seen_set.add((0, 0))
    
    dr_cur, dc_cur = 0, 0
    ans = []
    
    for i in range(N):
        c = S[i]
        if c == 'N':
            dr_cur -= 1
        elif c == 'S':
            dr_cur += 1
        elif c == 'W':
            dc_cur -= 1
        elif c == 'E':
            dc_cur += 1
            
        if c == 'N':
            state_req = (dr_cur - 1, dc_cur)
        elif c == 'S':
            state_req = (dr_cur + 1, dc_cur)
        elif c == 'W':
            state_req = (dr_cur, dc_cur - 1)
        elif c == 'E':
            state_req = (dr_cur, dc_cur + 1)
        
        if state_req in seen_set:
            gen_this = False
        else:
            gen_this = True
            
        seen_set.add((dr_cur, dc_cur))
        
        X = (dr_cur - R, dc_cur - C)
        if X in generated_set:
            ans.append('1')
        else:
            ans.append('0')
            
        if gen_this:
            generated_set.add((dr_cur, dc_cur))
            
    print(''.join(ans))

if __name__ == "__main__":
    main()