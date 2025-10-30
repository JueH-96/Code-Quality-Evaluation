def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))

    def check(min_mass):
        extended_a = a * 2
        for start in range(n):
            num_groups = 0
            current_start = start
            for _ in range(k):
                found_group = False
                current_group_sum = 0
                for end in range(current_start, start + n):
                    current_group_sum += extended_a[end % n]
                    if current_group_sum >= min_mass:
                        current_start = end + 1
                        num_groups += 1
                        found_group = True
                        break
                if not found_group:
                    break
            if num_groups == k:
                return True
        return False

    left, right = min(a), sum(a)
    max_min_mass = 0
    while left <= right:
        mid = (left + right) // 2
        if check(mid):
            max_min_mass = mid
            left = mid + 1
        else:
            right = mid - 1

    def find_cuts(division):
        cuts = set()
        ptr = 0
        for group in division:
            end_index = ptr + len(group)
            if end_index < n:
                cuts.add(end_index)
            elif end_index > n:
                cuts.add(end_index % n)
            ptr = end_index % n
        return cuts

    def get_all_valid_divisions(min_mass):
        valid_divisions = []

        def find_divisions(current_division, remaining_pieces, num_remaining_groups):
            if num_remaining_groups == 0:
                if not remaining_pieces:
                    valid_divisions.append(current_division)
                return

            if not remaining_pieces:
                return

            current_sum = 0
            for i in range(len(remaining_pieces)):
                current_sum += remaining_pieces[i]
                if current_sum >= min_mass:
                    find_divisions(current_division + [remaining_pieces[:i+1]], remaining_pieces[i+1:], num_remaining_groups - 1)

        extended_a = a * 2
        for start_index in range(n):
            def find_divisions_circular(current_division, current_index, num_remaining_groups, start_origin):
                if num_remaining_groups == 0:
                    if current_index == start_origin + n:
                        valid_divisions.append(current_division)
                    return

                if current_index >= start_origin + n:
                    return

                current_sum = 0
                for end_offset in range(1, n + 1):
                    end_index = current_index + end_offset
                    group_indices = [(current_index + i) % n for i in range(end_offset)]
                    group_sum = sum(a[i] for i in group_indices)

                    if group_sum >= min_mass:
                        find_divisions_circular(current_division + [tuple(sorted(group_indices))], current_index + end_offset, num_remaining_groups - 1, start_origin)

            find_divisions_circular([], start_index, k, start_index)

        return valid_divisions

    # This part is complex and might need a different approach to count uncut lines efficiently.
    # Let's try a simpler approach for counting uncut lines.

    never_cut_lines = set(range(1, n + 1))
    for cut_line in range(1, n + 1):
        can_be_uncut = False
        extended_a = a * 2
        for start in range(n):
            def check_uncut(division_cuts):
                return cut_line not in division_cuts

            def find_division_with_uncut(current_division_indices, num_remaining_groups, current_start_index):
                nonlocal can_be_uncut
                if num_remaining_groups == 0:
                    can_be_uncut = True
                    return

                if can_be_uncut:
                    return

                for end_offset in range(1, n + 1):
                    end_index = current_start_index + end_offset
                    group_indices = tuple(sorted([(current_start_index + i) % n for i in range(end_offset)]))
                    group_sum = sum(a[i] for i in group_indices)

                    if group_sum >= max_min_mass:
                        next_start = current_start_index + end_offset
                        if num_remaining_groups == 1:
                            if next_start == start + n:
                                # Check if cut_line is not cut
                                is_cut = False
                                for i in range(len(current_division_indices)):
                                    group1_end = current_division_indices[i][-1]
                                    group2_start = current_division_indices[i+1][0] if i + 1 < len(current_division_indices) else group_indices[0]

                                    if (group1_end + 1) % n == group2_start:
                                        if (group1_end + 1) == cut_line or group1_end + 1 == (cut_line % n) + (n if cut_line % n == 0 else 0):
                                            is_cut = True
                                            break
                                if not is_cut:
                                    can_be_uncut = True
                                return

                        find_division_with_uncut(current_division_indices + [group_indices], num_remaining_groups - 1, next_start)

            # Try to form a division where cut_line is not cut
            p1 = (cut_line - 1) % n
            p2 = cut_line % n

            def find_division_with_pair(current_division_indices, num_remaining_groups, current_start_index):
                nonlocal can_be_uncut
                if num_remaining_groups == 0:
                    can_be_uncut = True
                    return

                if can_be_uncut:
                    return

                for end_offset in range(1, n + 1):
                    end_index = current_start_index + end_offset
                    group_indices = tuple(sorted([(current_start_index + i) % n for i in range(end_offset)]))
                    group_sum = sum(a[i] for i in group_indices)

                    if group_sum >= max_min_mass:
                        next_start = current_start_index + end_offset
                        find_division_with_pair(current_division_indices + [group_indices], num_remaining_groups - 1, next_start)

            # Try forming K groups where pieces p1 and p2 are together
            for first_group_size in range(1, n + 1):
                first_group_indices = tuple(sorted([i % n for i in range(start, start + first_group_size)]))
                if p1 in first_group_indices and p2 in first_group_indices:
                    first_group_sum = sum(a[i] for i in first_group_indices)
                    if first_group_sum >= max_min_mass:
                        def find_remaining(current_division, num_remaining, current_start):
                            nonlocal can_be_uncut
                            if num_remaining == 0:
                                can_be_uncut = True
                                return
                            if can_be_uncut:
                                return

                            for size in range(1, n + 1):
                                indices = tuple(sorted([(current_start + i) % n for i in range(size)]))
                                s = sum(a[i] for i in indices)
                                if s >= max_min_mass:
                                    find_remaining(current_division + [indices], num_remaining - 1, current_start + size)

                        find_remaining([first_group_indices], k - 1, start + first_group_size)

        if not can_be_uncut:
            never_cut_lines.discard(cut_line)

    print(max_min_mass, len(never_cut_lines))

solve()