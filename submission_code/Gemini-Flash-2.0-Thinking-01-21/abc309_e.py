def solve():
    n, m = map(int, input().split())
    if n == 1:
        parents = []
    else:
        parents = list(map(int, input().split()))
    insurances = []
    for _ in range(m):
        insurances.append(list(map(int, input().split())))
    
    children_list = [[] for _ in range(n + 1)]
    for i in range(2, n + 1):
        parent = parents[i-2]
        children_list[parent].append(i)
        
    total_covered_people = set()
    
    for insurance in insurances:
        x_i, y_i = insurance
        current_insurance_coverage = {x_i}
        queue = [(x_i, 0)]
        while queue:
            person, generation = queue.pop(0)
            if generation < y_i:
                for child in children_list[person]:
                    if child not in current_insurance_coverage:
                        current_insurance_coverage.add(child)
                        queue.append((child, generation + 1))
        total_covered_people.update(current_insurance_coverage)
        
    print(len(total_covered_people))

if __name__ == '__main__':
    solve()