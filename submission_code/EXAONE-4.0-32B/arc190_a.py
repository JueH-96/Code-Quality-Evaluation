import sys

def main():
    data = sys.stdin.read().strip().split()
    if not data:
        return
    
    n = int(data[0])
    m = int(data[1])
    intervals = []
    index = 2
    for i in range(m):
        l = int(data[index]); r = int(data[index+1]); index += 2
        intervals.append((l, r, i))
    
    for i in range(m):
        l, r, idx = intervals[i]
        if l == 1 and r == n:
            ops = [0] * m
            ops[i] = 1
            print(1)
            print(" ".join(map(str, ops)))
            return
            
    sorted_intervals = sorted(intervals, key=lambda x: (x[0], -x[1]))
    current = 0
    count = 0
    chosen_indices = []
    for i in range(len(sorted_intervals)):
        if count >= 2:
            break
        l, r, orig_idx = sorted_intervals[i]
        if l <= current + 1:
            if r > current:
                current = r
                count += 1
                chosen_indices.append(orig_idx)
        else:
            break
            
    if current >= n and count <= 2:
        ops = [0] * m
        for idx in chosen_indices:
            ops[idx] = 1
        print(count)
        print(" ".join(map(str, ops)))
        return
        
    sorted_intervals2 = sorted(intervals, key=lambda x: x[0])
    min_r = 10**9
    min_r_idx = -1
    found_pair = None
    for i in range(len(sorted_intervals2)):
        l, r, orig_idx = sorted_intervals2[i]
        if min_r < l:
            found_pair = (min_r_idx, i)
            break
        if r < min_r:
            min_r = r
            min_r_idx = i
            
    if found_pair is not None:
        idx1, idx2 = found_pair
        _, _, orig_idx1 = sorted_intervals2[idx1]
        _, _, orig_idx2 = sorted_intervals2[idx2]
        ops = [0] * m
        ops[orig_idx1] = 2
        ops[orig_idx2] = 2
        print(2)
        print(" ".join(map(str, ops)))
        return
        
    print(-1)

if __name__ == "__main__":
    main()