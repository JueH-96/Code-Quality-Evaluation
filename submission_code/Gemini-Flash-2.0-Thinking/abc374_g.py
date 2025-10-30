def solve():
    n = int(input())
    used_names = set()
    for _ in range(n):
        used_names.add(input())

    covered_names = set()
    ng_list_count = 0

    while len(covered_names) < n:
        ng_list_count += 1
        uncovered_name = next(iter(used_names - covered_names))

        ng_string = uncovered_name
        covered_names.add(uncovered_name)

        # Extend forward
        while True:
            extended = False
            last_char = ng_string[-1]
            for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                potential_extension = last_char + char
                if potential_extension in used_names and potential_extension not in covered_names:
                    ng_string += char
                    covered_names.add(potential_extension)
                    extended = True
                    break
            if not extended:
                break

        # Extend backward
        while True:
            extended = False
            first_char = ng_string[0]
            for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                potential_extension = char + first_char
                if potential_extension in used_names and potential_extension not in covered_names:
                    ng_string = char + ng_string
                    covered_names.add(potential_extension)
                    extended = True
                    break
            if not extended:
                break

    print(ng_list_count)

solve()