# YOUR CODE HERE
import sys

def main():
    import sys
    M = int(sys.stdin.readline())
    S = [sys.stdin.readline().strip() for _ in range(3)]
    
    T_max = M * 30  # assuming k up to 30
    lists = [{} for _ in range(3)]
    
    for i in range(3):
        for c in '0123456789':
            lists[i][c] = []
        for k in range(0, T_max +1):
            pos = k % M
            c = S[i][pos]
            if k not in lists[i][c]:
                lists[i][c].append(k)
    
    min_T = float('inf')
    for c in '0123456789':
        l0 = lists[0].get(c, [])
        l1 = lists[1].get(c, [])
        l2 = lists[2].get(c, [])
        if not l0 or not l1 or not l2:
            continue
        for t0 in l0:
            for t1 in l1:
                if t1 == t0:
                    continue
                for t2 in l2:
                    if t2 == t0 or t2 == t1:
                        continue
                    T_candidate = max(t0, t1, t2)
                    if T_candidate < min_T:
                        min_T = T_candidate
    if min_T == float('inf'):
        print(-1)
    else:
        print(min_T)

if __name__ == "__main__":
    main()