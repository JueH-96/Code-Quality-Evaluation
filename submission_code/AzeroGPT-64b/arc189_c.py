# Solution for the provided problem

from collections import defaultdict


def read_int_list():
    return [int(item) for item in input().split()]


# def dfs(p, q, visited, from_index):

#     visited[from_index] = True
#     #print(f"Visiting {from_index}", visited, p[from_index], q[from_index], p, q)

#     if p[from_index] not in visited:
#         visited = dfs(p, q, visited, p[from_index] - 1)

#     if q[from_index] not in visited:
#         visited = dfs(p, q, visited, q[from_index] - 1)

#     return visited


def cycle(p, element, indices, count, cycle_start):
    sequence = []
    i = element

    while i not in indices:
        cycle_vertex = p[i]
        sequence.append((i, cycle_vertex))
        indices[i] = (True, count, cycle_start)

        if cycle_vertex in indices:
            break
        i = cycle_vertex

    # Close cycle
    if indices[i][2] == cycle_start and i != cycle_start:
        for cycle_vertex, _ in sequence:
            indices[cycle_vertex][0] = False
    return count+1


def main():
    n, x = read_int_list()
    a = read_int_list()
    b = read_int_list()
    p = read_int_list()
    q = read_int_list()

    indices = defaultdict(list)
    start = [0]*(n+1)
    count = 1

    for i in range(n):
        if start[i+1]:
            continue

        count = cycle(p, i, indices, count, i)
        start[i+1] = True

    nodes_to_revisit = defaultdict(list)
    for i in range(len(p)):
        nodes_to_revisit[p[i]-1].append(i+1)
        nodes_to_revisit[q[i]-1].append(i+1)
    
    answer = [0]*(n+1)

    def bad_position(position_to_check):
        for idx, elem in indices.items():
            if answer[idx] == 0 and not elem[0] and idx == position_to_check:
                return True
        return False

    ball_allocation = {(0,0) : 0, 0: 3}
    for i in range(1, n+1):
        if answer[i]:
            continue
        if x == i or a[i-1] + b[i-1] == 0:
            answer[i] = 2
            continue

        u, v = 0, 0
        if a[i-1] == 1:
            u = 1
        if b[i-1]:
            v = 1
        allocation = ball_allocation[(u, v)]


        if allocation == 0:
            if bad_position(i):
                print(-1)
                return

            if len(nodes_to_revisit[i]) == 2 and  (nodes_to_revisit[i][0] in indices and nodes_to_revisit[i][1] in indices) and indices[nodes_to_revisit[i][0]][0] and indices[nodes_to_revisit[i][1]][0]:
                answer[i] = 3
                continue

            if len(nodes_to_revisit[i]) == 2:
                for neighbor in nodes_to_revisit[i]:
                    if neighbor == x and a[neighbor-1] + b[neighbor-1]:
                        answer[i] = 2
                        continue
                    if nodes_to_revisit[neighbor] != [i]:
                        answer[i] = 1

            if len(nodes_to_revisit[i]) == 1:
                answer[i] = 1

            if answer[i] == 0:
                answer[i] = 2

        else:
            if allocation != 3 and len(nodes_to_revisit[i]) == 2 and  (nodes_to_revisit[i][0] in indices and nodes_to_revisit[i][1] in indices) and indices[nodes_to_revisit[i][0]][0] and indices[nodes_to_revisit[i][1]][0]:                    
                print(-1)
                return

    # print(indices)
    # print("balls", a, b)
    # print("p", p)
    # print("q", q)
    # print(nodes_to_revisit)
    # print(answer)
   
    if answer[x] != 3:
        print(-1)
        return

    print(sum([item for item in answer if item == 1 or item == 2]))


if __name__ == "__main__":
    main()