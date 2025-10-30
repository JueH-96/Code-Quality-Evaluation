def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N, T, P = map(int, data[:3])
    L = list(map(int, data[3:]))

    # Count how many already have hair length >= T
    count_now = sum(1 for x in L if x >= T)
    if count_now >= P:
        print(0)
        return

    # Calculate how many days each person needs to reach at least T
    days_needed = [max(0, T - x) for x in L]

    # Sort the required days in ascending order
    days_needed.sort()

    # We already have count_now people,
    # so we only need to add (P - count_now) more from the sorted list
    needed = P - count_now
    day = 0
    i = 0
    # Move through days_needed from smallest to largest
    while needed > 0 and i < N:
        if L[i] < T:  # This person is not counted in count_now
            day = days_needed[i]
            needed -= 1
        i += 1

    print(day)

# Let's call solve() to run the solution
def main():
    solve()

if __name__ == "__main__":
    main()