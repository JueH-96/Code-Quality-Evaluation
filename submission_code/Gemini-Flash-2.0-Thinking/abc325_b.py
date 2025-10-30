def solve():
    n = int(input())
    bases = []
    for _ in range(n):
        w, x = map(int, input().split())
        bases.append((w, x))

    max_employees = 0
    for utc_start in range(24):
        current_employees = 0
        for w, x in bases:
            local_start = (utc_start + x) % 24
            local_end = (utc_start + x + 1) % 24

            can_participate = False
            if local_start < local_end:
                if 9 <= local_start <= 17 and 9 <= local_end <= 18:
                    can_participate = True
            elif local_start > local_end:
                # Meeting crosses midnight, cannot be within 9-18
                pass
            else:  # local_start == local_end, impossible for 1-hour meeting
                pass

            if can_participate:
                current_employees += w

        max_employees = max(max_employees, current_employees)

    print(max_employees)

solve()