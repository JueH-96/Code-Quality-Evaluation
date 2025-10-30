import sys

def main():
    data = sys.stdin.read().split()
    t = int(data[0])
    index = 1
    results = []
    for _ in range(t):
        n = int(data[index]); index += 1
        total_len = 2 * n
        A = list(map(int, data[index:index+total_len]))
        index += total_len
        
        if n == 0:
            results.append(0)
            continue
            
        pos_list = [[] for _ in range(n+1)]
        for idx, num in enumerate(A):
            if 1 <= num <= n:
                pos_list[num].append(idx)
                
        adjacent = [False] * (n+1)
        for i in range(1, n+1):
            if len(pos_list[i]) < 2:
                adjacent[i] = False
            else:
                p0, p1 = pos_list[i]
                if p1 - p0 == 1:
                    adjacent[i] = True
                else:
                    adjacent[i] = False
                    
        other_pos = [-1] * total_len
        for i in range(1, n+1):
            if len(pos_list[i]) == 2:
                p0, p1 = pos_list[i]
                other_pos[p0] = p1
                other_pos[p1] = p0
                
        candidate_set = set()
        
        for i in range(total_len-1):
            a_val = A[i]
            b_val = A[i+1]
            if a_val == b_val:
                continue
            if a_val < 1 or a_val > n or b_val < 1 or b_val > n:
                continue
            if adjacent[a_val] or adjacent[b_val]:
                continue
                
            j = other_pos[i]
            k = other_pos[i+1]
            if j == -1 or k == -1:
                continue
                
            if abs(j - k) == 1:
                j0 = min(j, k)
                if (i+1 < j0) or (j0+1 < i):
                    if a_val < b_val:
                        pair = (a_val, b_val)
                    else:
                        pair = (b_val, a_val)
                    candidate_set.add(pair)
                    
        results.append(len(candidate_set))
        
    for res in results:
        print(res)

if __name__ == "__main__":
    main()