import sys

def solve():
    n, k = map(int, sys.stdin.readline().split())
    a = list(map(int, sys.stdin.readline().split()))
    
    def get_groups_and_min_sum(cut_lines_indices):
        cut_lines_set = set(cut_lines_indices)
        groups = []
        current_group = []
        for i in range(n):
            current_group.append(i)
            if (i + 1) % n in cut_lines_set:
                groups.append(current_group)
                current_group = []
        if current_group:
            first_group = groups[0]
            groups = groups[1:]
            groups.append(first_group + current_group)
            
        group_masses = []
        for group_indices in groups:
            group_mass = sum(a[i] for i in group_indices)
            group_masses.append(group_mass)
        return groups, min(group_masses) if group_masses else 0

    def get_uncut_lines(groups):
        uncut_lines = set()
        for i in range(n):
            piece1_index = i
            piece2_index = (i + 1) % n
            group1_index = -1
            group2_index = -1
            for gi in range(len(groups)):
                if piece1_index in groups[gi]:
                    group1_index = gi
                if piece2_index in groups[gi]:
                    group2_index = gi
            if group1_index == group2_index:
                uncut_lines.add(i + 1)
        return uncut_lines

    max_min_mass = 0
    optimal_divisions_uncut_lines = []

    import itertools
    
    cut_line_indices_options = itertools.combinations(range(n), k)
    
    for cut_line_indices in cut_line_indices_options:
        groups, min_sum = get_groups_and_min_sum(cut_line_indices)
        if min_sum > max_min_mass:
            max_min_mass = min_sum
            optimal_divisions_uncut_lines = [get_uncut_lines(groups)]
        elif min_sum == max_min_mass:
            optimal_divisions_uncut_lines.append(get_uncut_lines(groups))

    common_uncut_lines = set(range(1, n + 1))
    if optimal_divisions_uncut_lines:
        common_uncut_lines = optimal_divisions_uncut_lines[0]
        for i in range(1, len(optimal_divisions_uncut_lines)):
            common_uncut_lines = common_uncut_lines.intersection(optimal_divisions_uncut_lines[i])

    print(max_min_mass, len(common_uncut_lines))

if __name__ == '__main__':
    solve()