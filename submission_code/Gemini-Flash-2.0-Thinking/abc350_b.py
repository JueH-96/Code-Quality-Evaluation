def solve():
    n, q = map(int, input().split())
    treatments = list(map(int, input().split()))

    teeth = set(range(1, n + 1))

    for treatment in treatments:
        if treatment in teeth:
            teeth.remove(treatment)
        else:
            teeth.add(treatment)

    print(len(teeth))

# YOUR CODE HERE
solve()