import bisect

def get_subset_sums(arr):
    n = len(arr)
    sums = []
    for mask in range(1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += arr[i]
        sums.append(s)
    return sums

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    K = list(map(int, input[1:N+1]))
    
    mid = N // 2
    first_half = K[:mid]
    second_half = K[mid:]
    
    sumA_list = get_subset_sums(first_half)
    sumB_list = get_subset_sums(second_half)
    
    sumB_list.sort()
    
    S = sum(K)
    min_max = float('inf')
    
    for sumA in sumA_list:
        target = S / 2 - sumA
        idx = bisect.bisect_left(sumB_list, target)
        candidates = []
        if idx < len(sumB_list):
            candidates.append(sumB_list[idx])
        if idx > 0:
            candidates.append(sumB_list[idx - 1])
        for sumB in candidates:
            sum_total = sumA + sumB
            current_max = max(sum_total, S - sum_total)
            if current_max < min_max:
                min_max = current_max
                
    print(int(min_max))

if __name__ == '__main__':
    main()