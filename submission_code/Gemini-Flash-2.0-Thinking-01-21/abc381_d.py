def solve():
    n = int(input())
    a = list(map(int, input().split()))
    max_length = 0
    for start_index in range(n):
        current_length = 0
        used_numbers = set()
        index = start_index
        while index < n - 1:
            if a[index] == a[index+1] and a[index] not in used_numbers:
                used_numbers.add(a[index])
                current_length += 2
                index += 2
            else:
                break
        max_length = max(max_length, current_length)
    print(max_length)

if __name__ == '__main__':
    solve()