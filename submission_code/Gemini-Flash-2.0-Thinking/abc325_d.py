def solve():
    n = int(input())
    products = []
    for _ in range(n):
        t, d = map(int, input().split())
        products.append((t, d))

    max_printed = 0

    import itertools

    for printed_indices in itertools.combinations(range(n), n):
        current_printed_count = 0
        last_print_time = -float('inf')

        can_print_all = True
        print_times = {}

        def find_earliest_possible_print_time(product_index, after_time):
            t, d = products[product_index]
            start_time = t
            end_time = t + d
            if start_time >= after_time + 1:
                return start_time
            elif end_time >= after_time + 1:
                return after_time + 1
            else:
                return -1

        def find_latest_possible_print_time(product_index):
            t, d = products[product_index]
            return t + d

        def can_print(print_time, after_time):
            return print_time >= after_time + 1

        def find_possible_print_times(product_index):
            t, d = products[product_index]
            return range(int(t), int(t + d) + 1)

        def check_schedule(printed_products_indices, print_schedule):
            if not printed_products_indices:
                return True

            sorted_prints = sorted(print_schedule.items(), key=lambda item: item[1])

            for i in range(len(sorted_prints)):
                if i > 0 and sorted_prints[i][1] <= sorted_prints[i-1][1]:
                    return False
                if i > 0 and sorted_prints[i][1] < sorted_prints[i-1][1] + 1:
                    return False

                product_index = sorted_prints[i][0]
                print_time = sorted_prints[i][1]
                t, d = products[product_index]
                if not (t <= print_time <= t + d):
                    return False
            return True

        for r in range(1, n + 1):
            for combination in itertools.combinations(range(n), r):
                possible = True
                schedule = {}
                last_printed_time = -float('inf')
                for prod_index in combination:
                    t, d = products[prod_index]
                    can_print_now = False
                    for print_time in range(int(t), int(t + d) + 1):
                        if print_time >= last_printed_time + 1:
                            schedule[prod_index] = print_time
                            last_printed_time = print_time
                            can_print_now = True
                            break
                    if not can_print_now:
                        possible = False
                        break
                if possible:
                    max_printed = max(max_printed, r)

    print(max_printed)

solve()