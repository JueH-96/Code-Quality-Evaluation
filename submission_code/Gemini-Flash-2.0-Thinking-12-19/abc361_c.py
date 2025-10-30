def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    sorted_a = sorted(list(set(a)))
    unique_sorted_a = sorted_a
    min_diff = float('inf')
    
    for i in range(len(unique_sorted_a)):
        for j in range(i, len(unique_sorted_a)):
            v_min = unique_sorted_a[i]
            v_max = unique_sorted_a[j]
            count = 0
            for val in a:
                if v_min <= val <= v_max:
                    count += 1
            if count >= n - k:
                min_diff = min(min_diff, v_max - v_min)
                
    print(min_diff)

if __name__ == '__main__':
    solve()