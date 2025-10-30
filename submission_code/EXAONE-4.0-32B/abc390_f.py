import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    
    last_occ = [0] * (n + 1)
    T_d = 0
    for i in range(n):
        val = A[i]
        prev = last_occ[val]
        T_d += (i + 1 - prev) * (n - i)
        last_occ[val] = i + 1
        
    total_segments = n * (n + 1) // 2
    
    positions = [[] for _ in range(n + 1)]
    for i in range(n):
        val = A[i]
        positions[val].append(i + 1)
        
    avoid = [0] * (n + 1)
    for v in range(1, n + 1):
        if not positions[v]:
            avoid[v] = total_segments
        else:
            gaps = []
            gaps.append(positions[v][0] - 1)
            for j in range(1, len(positions[v])):
                gap_val = positions[v][j] - positions[v][j - 1] - 1
                gaps.append(gap_val)
            gaps.append(n - positions[v][-1])
            total_avoid = 0
            for g in gaps:
                total_avoid += g * (g + 1) // 2
            avoid[v] = total_avoid
            
    T_c = 0
    for x in range(1, n):
        if not positions[x] or not positions[x + 1]:
            continue
        list1 = positions[x]
        list2 = positions[x + 1]
        i, j = 0, 0
        combined = []
        while i < len(list1) and j < len(list2):
            if list1[i] < list2[j]:
                combined.append(list1[i])
                i += 1
            else:
                combined.append(list2[j])
                j += 1
        while i < len(list1):
            combined.append(list1[i])
            i += 1
        while j < len(list2):
            combined.append(list2[j])
            j += 1
            
        gaps_comb = []
        gaps_comb.append(combined[0] - 1)
        for idx in range(1, len(combined)):
            gap_val = combined[idx] - combined[idx - 1] - 1
            gaps_comb.append(gap_val)
        gaps_comb.append(n - combined[-1])
        
        avoid_both = 0
        for g in gaps_comb:
            avoid_both += g * (g + 1) // 2
            
        count_both = total_segments - avoid[x] - avoid[x + 1] + avoid_both
        T_c += count_both
        
    ans = T_d - T_c
    print(ans)

if __name__ == "__main__":
    main()