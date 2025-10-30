# YOUR CODE HERE
def solve():
    n, t, m = map(int, input().split())
    incompatible_pairs = []
    for _ in range(m):
        incompatible_pairs.append(tuple(map(int, input().split())))

    def is_valid(assignment):
        for a, b in incompatible_pairs:
            if assignment[a - 1] == assignment[b - 1]:
                return False
        return True

    def count_valid_assignments():
        count = 0
        
        def generate_assignments(index, current_assignment):
            nonlocal count
            if index == n:
                if is_valid(current_assignment):
                    count += 1
                return

            for team in range(1, t + 1):
                generate_assignments(index + 1, current_assignment + [team])

        generate_assignments(0, [])
        return count

    print(count_valid_assignments())

solve()