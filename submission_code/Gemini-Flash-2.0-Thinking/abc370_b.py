def solve():
    n = int(input())
    a = []
    for _ in range(n):
        a.append(list(map(int, input().split())))

    current_element = 1
    for i in range(1, n + 1):
        element_to_combine = i
        if current_element >= element_to_combine:
            combined_element = a[current_element - 1][element_to_combine - 1]
        else:
            combined_element = a[element_to_combine - 1][current_element - 1]
        current_element = combined_element
    print(current_element)

solve()