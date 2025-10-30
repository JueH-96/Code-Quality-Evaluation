def solve():
    n = int(input())
    k = list(map(int, input().split()))
    min_max_people = float('inf')
    for i in range(2**n):
        group_a_people = 0
        group_b_people = 0
        binary_string = format(i, '0' + str(n) + 'b')
        for j in range(n):
            if binary_string[j] == '0':
                group_a_people += k[j]
            else:
                group_b_people += k[j]
        current_max_people = max(group_a_people, group_b_people)
        min_max_people = min(min_max_people, current_max_people)
    print(min_max_people)

if __name__ == '__main__':
    solve()