import sys
from collections import defaultdict, Counter

def main():
    data = sys.stdin.read().split()
    if not data:
        print(0.0)
        return
    
    n = int(data[0])
    index = 1
    dice = []
    for i in range(n):
        k = int(data[index])
        index += 1
        faces = list(map(int, data[index:index+k]))
        index += k
        dice.append((k, faces))
    
    global_dict = defaultdict(list)
    for idx, (k, faces) in enumerate(dice):
        cnt = Counter(faces)
        for num, count_val in cnt.items():
            prob = count_val / k
            global_dict[num].append((idx, prob))
            
    mat = [[0.0] * n for _ in range(n)]
    
    for num, lst in global_dict.items():
        m = len(lst)
        if m < 2:
            continue
        for i in range(m):
            idx_i, p_i = lst[i]
            for j in range(i+1, m):
                idx_j, p_j = lst[j]
                i1 = min(idx_i, idx_j)
                i2 = max(idx_i, idx_j)
                mat[i1][i2] += p_i * p_j
                
    best = 0.0
    for i in range(n):
        for j in range(i+1, n):
            if mat[i][j] > best:
                best = mat[i][j]
                
    print("{:.15f}".format(best))

if __name__ == "__main__":
    main()