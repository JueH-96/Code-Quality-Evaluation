# YOUR CODE HERE
def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, T, P = map(int, data[:3])
    L = list(map(int, data[3:]))

    # Count how many already satisfy L_i >= T
    c = sum(1 for x in L if x >= T)

    # If we already have at least P people with L_i >= T
    if c >= P:
        print(0)
        return

    # Otherwise, compute the days needed for each remaining person to reach T
    needed_days = [T - x for x in L if x < T]
    needed_days.sort()

    # The day we need is the (P-c)-th smallest in the sorted list of needed days
    ans = needed_days[P - c - 1]
    print(ans)

# Do not forget to call main()!
main()