def solve():
    n, m = map(int, input().split())
    if n > 1:
        parents = list(map(int, input().split()))
    else:
        parents = []
    insurances = []
    for _ in range(m):
        insurances.append(list(map(int, input().split())))

    children = [[] for _ in range(n)]
    for i, parent in enumerate(parents):
        children[parent - 1].append(i + 2)

    covered_people = set()

    for x, y in insurances:
        start_person = x
        generations = y

        queue = [(start_person, 0)]
        visited = {start_person}
        covered_by_current_insurance = {start_person}

        while queue:
            current_person, current_gen = queue.pop(0)

            if current_gen < generations:
                for child in children[current_person - 1]:
                    if child not in visited:
                        visited.add(child)
                        covered_by_current_insurance.add(child)
                        queue.append((child, current_gen + 1))

        covered_people.update(covered_by_current_insurance)

    print(len(covered_people))

solve()