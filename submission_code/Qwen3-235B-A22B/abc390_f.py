import sys
from collections import defaultdict

def compute_CA(list_p, N):
    if not list_p:
        return N * (N + 1) // 2
    prev = 0
    sum_ab = 0
    for p in list_p:
        gap_start = prev + 1
        gap_end = p - 1
        if gap_start <= gap_end:
            length = gap_end - gap_start + 1
            sum_ab += (length * (length + 1)) // 2
        prev = p
    gap_start = prev + 1
    gap_end = N
    if gap_start <= gap_end:
        length = gap_end - gap_start + 1
        sum_ab += (length * (length + 1)) // 2
    return sum_ab

def compute_CA_merged(merged, N):
    if not merged:
        return N * (N + 1) // 2
    prev = 0
    sum_ab = 0
    for p in merged:
        gap_start = prev + 1
        gap_end = p - 1
        if gap_start <= gap_end:
            length = gap_end - gap_start + 1
            sum_ab += (length * (length + 1)) // 2
        prev = p
    gap_start = prev + 1
    gap_end = N
    if gap_start <= gap_end:
        length = gap_end - gap_start + 1
        sum_ab += (length * (length + 1)) // 2
    return sum_ab

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    # Compute sum_distinct
    sum_distinct = 0
    last = dict()
    for i in range(N):
        a = A[i]
        prev_pos = last.get(a, 0)
        cnt = (i + 1 - prev_pos) * (N - i)
        sum_distinct += cnt
        last[a] = i + 1  # 1-based index
    
    # Preprocess positions
    pos = defaultdict(list)
    for i in range(N):
        a = A[i]
        pos[a].append(i + 1)  # 1-based
    
    total_intervals = N * (N + 1) // 2
    sum_consecutive = 0
    
    for x in range(1, N):
        y = x + 1
        list_x = pos.get(x, [])
        list_y = pos.get(y, [])
        C_Ax = compute_CA(list_x, N)
        C_Ay = compute_CA(list_y, N)
        merged = sorted(list_x + list_y)
        C_AB = compute_CA_merged(merged, N)
        Fx = total_intervals - C_Ax - C_Ay + C_AB
        sum_consecutive += Fx
    
    answer = sum_distinct - sum_consecutive
    print(answer)

if __name__ == "__main__":
    main()