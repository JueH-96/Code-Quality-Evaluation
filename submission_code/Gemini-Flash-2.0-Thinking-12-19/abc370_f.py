import sys

def solve():
    n, k = map(int, sys.stdin.readline().split())
    a = list(map(int, sys.stdin.readline().split()))
    
    def get_group_masses(cut_lines_indices):
        cut_lines_set = set(cut_lines_indices)
        groups = []
        current_group = []
        for i in range(n):
            current_group.append(i)
            cut_line_index = (i + 1) if (i < n - 1) else 0
            if cut_line_index in cut_lines_set:
                groups.append(current_group)
                current_group = []
        if current_group:
            groups.append(current_group)
        group_masses = []
        for group in groups:
            mass = sum(a[i] for i in group)
            group_masses.append(mass)
        return group_masses

    max_min_mass = 0
    optimal_cut_line_sets = []
    
    import itertools
    
    cut_line_indices_options = list(range(n))
    all_cut_line_combinations = list(itertools.combinations(cut_line_indices_options, k))
    
    for cut_lines_tuple in all_cut_line_combinations:
        cut_lines_indices = list(cut_lines_tuple)
        group_masses = get_group_masses(cut_lines_indices)
        if len(group_masses) != k:
            continue
        min_mass = min(group_masses)
        if min_mass > max_min_mass:
            max_min_mass = min_mass
            optimal_cut_line_sets = [set(cut_lines_indices)]
        elif min_mass == max_min_mass:
            optimal_cut_line_sets.append(set(cut_lines_indices))
            
    never_cut_lines = set(range(1, n + 1))
    first_set = True
    
    for cut_set in optimal_cut_line_sets:
        current_uncut_lines = set()
        for i in range(n):
            if i not in cut_set:
                current_uncut_lines.add(i + 1)
        if first_set:
            never_cut_lines = current_uncut_lines
            first_set = False
        else:
            never_cut_lines = never_cut_lines.intersection(current_uncut_lines)
            
    print(max_min_mass, len(never_cut_lines))

if __name__ == '__main__':
    solve()