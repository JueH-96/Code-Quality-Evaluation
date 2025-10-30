def solve():
    n = int(input())
    a = list(map(int, input().split()))
    total_sum = 0
    last_occurrence_index = {}
    for k in range(1, n + 1):
        current_value = a[k-1]
        p = 0
        if current_value in last_occurrence_index:
            p = last_occurrence_index[current_value]
        contribution = (k - p) * (n - k + 1)
        total_sum += contribution
        last_occurrence_index[current_value] = k
    print(total_sum)

if __name__ == '__main__':
    solve()