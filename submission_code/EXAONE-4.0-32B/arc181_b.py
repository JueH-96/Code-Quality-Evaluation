import sys
import bisect

def main():
    data = sys.stdin.read().splitlines()
    t = int(data[0])
    index = 1
    out_lines = []
    
    base1 = 131
    base2 = 131
    mod1 = 10**9 + 7
    mod2 = 10**9 + 9
    
    for _ in range(t):
        S = data[index].strip()
        index += 1
        X = data[index].strip()
        index += 1
        Y = data[index].strip()
        index += 1
        
        n0x = X.count('0')
        n1x = len(X) - n0x
        n0y = Y.count('0')
        n1y = len(Y) - n0y
        n = len(S)
        
        if n1x == n1y:
            if n0x == n0y:
                out_lines.append("Yes")
                continue
            else:
                out_lines.append("No")
                continue
                
        num_val = (n0x - n0y) * n
        den_val = (n1y - n1x)
        
        if den_val == 0:
            out_lines.append("No")
            continue
            
        if num_val % den_val != 0:
            out_lines.append("No")
            continue
            
        LT = num_val // den_val
        if LT < 0:
            out_lines.append("No")
            continue
            
        boundaries_X = [0]
        for c in X:
            add = n if c == '0' else LT
            new_bound = boundaries_X[-1] + add
            boundaries_X.append(new_bound)
            
        boundaries_Y = [0]
        for c in Y:
            add = n if c == '0' else LT
            new_bound = boundaries_Y[-1] + add
            boundaries_Y.append(new_bound)
            
        if boundaries_X[-1] != boundaries_Y[-1]:
            out_lines.append("No")
            continue
            
        events = set(boundaries_X) | set(boundaries_Y)
        events = sorted(events)
        segments = []
        for j in range(len(events)-1):
            if events[j] != events[j+1]:
                segments.append((events[j], events[j+1]))
                
        if n > 0:
            prefix1 = [0] * (n+1)
            pow1 = [1] * (n+1)
            for j in range(n):
                prefix1[j+1] = (prefix1[j] * base1 + ord(S[j])) % mod1
                pow1[j+1] = (pow1[j] * base1) % mod1
                
            prefix2 = [0] * (n+1)
            pow2 = [1] * (n+1)
            for j in range(n):
                prefix2[j+1] = (prefix2[j] * base2 + ord(S[j])) % mod2
                pow2[j+1] = (pow2[j] * base2) % mod2
                
            def get_hash(l, r, prefix, power, base, mod):
                res = (prefix[r+1] - prefix[l] * power[r-l+1]) % mod
                return res if res >= 0 else res + mod
        else:
            def get_hash(l, r, prefix, power, base, mod):
                return 0
                
        valid = True
        constraints = []
        
        for (p, q) in segments:
            L_seg = q - p
            idx_x = bisect.bisect_right(boundaries_X, p) - 1
            block_x = X[idx_x]
            start_in_block_x = boundaries_X[idx_x]
            
            idx_y = bisect.bisect_right(boundaries_Y, p) - 1
            block_y = Y[idx_y]
            start_in_block_y = boundaries_Y[idx_y]
            
            if block_x == '0' and block_y == '0':
                a1 = p - start_in_block_x
                a2 = p - start_in_block_y
                if a1 < 0 or a1 + L_seg > n or a2 < 0 or a2 + L_seg > n:
                    valid = False
                    break
                h11 = get_hash(a1, a1+L_seg-1, prefix1, pow1, base1, mod1)
                h12 = get_hash(a1, a1+L_seg-1, prefix2, pow2, base2, mod2)
                h21 = get_hash(a2, a2+L_seg-1, prefix1, pow1, base1, mod1)
                h22 = get_hash(a2, a2+L_seg-1, prefix2, pow2, base2, mod2)
                if (h11, h12) != (h21, h22):
                    valid = False
                    break
            elif block_x == '0' and block_y == '1':
                a = p - start_in_block_x
                b = p - start_in_block_y
                if a < 0 or a + L_seg > n:
                    valid = False
                    break
                constraints.append((b, L_seg, a))
            elif block_x == '1' and block_y == '0':
                a = p - start_in_block_y
                b = p - start_in_block_x
                if a < 0 or a + L_seg > n:
                    valid = False
                    break
                constraints.append((b, L_seg, a))
            else:
                pass
                
        if not valid:
            out_lines.append("No")
            continue
            
        constraints.sort(key=lambda x: x[0])
        ncon = len(constraints)
        found_conflict = False
        for i in range(ncon):
            b1, L1, a1 = constraints[i]
            e1 = b1 + L1
            for j in range(i+1, ncon):
                b2, L2, a2 = constraints[j]
                e2 = b2 + L2
                overlap_start = max(b1, b2)
                overlap_end = min(e1, e2)
                if overlap_start >= overlap_end:
                    continue
                overlap_len = overlap_end - overlap_start
                s1_start = a1 + (overlap_start - b1)
                s1_end = s1_start + overlap_len
                s2_start = a2 + (overlap_start - b2)
                s2_end = s2_start + overlap_len
                if s1_start < 0 or s1_end > n or s2_start < 0 or s2_end > n:
                    found_conflict = True
                    break
                h11 = get_hash(s1_start, s1_end-1, prefix1, pow1, base1, mod1)
                h12 = get_hash(s1_start, s1_end-1, prefix2, pow2, base2, mod2)
                h21 = get_hash(s2_start, s2_end-1, prefix1, pow1, base1, mod1)
                h22 = get_hash(s2_start, s2_end-1, prefix2, pow2, base2, mod2)
                if (h11, h12) != (h21, h22):
                    found_conflict = True
                    break
            if found_conflict:
                break
                
        if found_conflict:
            out_lines.append("No")
        else:
            out_lines.append("Yes")
            
    for ans in out_lines:
        print(ans)

if __name__ == "__main__":
    main()