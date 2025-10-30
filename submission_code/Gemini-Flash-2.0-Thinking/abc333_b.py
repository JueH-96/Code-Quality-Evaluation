def solve():
    s1s2_str = input()
    t1t2_str = input()

    s1 = s1s2_str[0]
    s2 = s1s2_str[1]
    t1 = t1t2_str[0]
    t2 = t1t2_str[1]

    def vertex_to_num(vertex):
        mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
        return mapping[vertex]

    def calculate_distance(v1, v2):
        n1 = vertex_to_num(v1)
        n2 = vertex_to_num(v2)
        diff = abs(n1 - n2)
        return min(diff, 5 - diff)

    dist1 = calculate_distance(s1, s2)
    dist2 = calculate_distance(t1, t2)

    if dist1 == dist2:
        print("Yes")
    else:
        print("No")

solve()