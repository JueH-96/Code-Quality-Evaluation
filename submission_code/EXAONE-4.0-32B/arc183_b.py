import sys
from collections import defaultdict

def main():
    data = sys.stdin.read().split()
    t = int(data[0])
    index = 1
    out_lines = []
    for _ in range(t):
        n = int(data[index]); index += 1
        k = int(data[index]); index += 1
        A = list(map(int, data[index:index+n])); index += n
        B = list(map(int, data[index:index+n])); index += n
        
        if A == B:
            out_lines.append("Yes")
            continue
            
        setA = set(A)
        setB = set(B)
        if not setB.issubset(setA):
            out_lines.append("No")
            continue
            
        occ_dict = defaultdict(set)
        req_dict = defaultdict(set)
        for i in range(n):
            occ_dict[A[i]].add(i)
        for i in range(n):
            req_dict[B[i]].add(i)
            
        values_in_B = set(B)
        valid = True
        for x in values_in_B:
            occ_set = occ_dict[x]
            req_set = req_dict[x]
            events_asc = sorted(occ_set | req_set)
            covered_by_left = set()
            cover_to = -10**18
            for pos in events_asc:
                if pos in occ_set:
                    if pos + k > cover_to:
                        cover_to = pos + k
                if pos in req_set:
                    if pos <= cover_to:
                        covered_by_left.add(pos)
                        
            events_desc = sorted(occ_set | req_set, reverse=True)
            cover_from = 10**18
            found_failure = False
            for pos in events_desc:
                if pos in occ_set:
                    if pos - k < cover_from:
                        cover_from = pos - k
                if pos in req_set:
                    if pos not in covered_by_left:
                        if pos < cover_from:
                            found_failure = True
                            break
            if found_failure:
                out_lines.append("No")
                valid = False
                break
                
        if valid:
            out_lines.append("Yes")
            
    sys.stdout.write("
".join(out_lines))

if __name__ == "__main__":
    main()