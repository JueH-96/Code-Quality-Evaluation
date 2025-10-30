def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # dp_even: max XP if we have defeated an even number of monsters so far
    # dp_odd: max XP if we have defeated an odd number of monsters so far
    dp_even, dp_odd = 0, float('-inf')

    for x in A:
        new_even = max(dp_even, dp_odd + 2*x)
        new_odd = max(dp_odd, dp_even + x)
        dp_even, dp_odd = new_even, new_odd

    print(max(dp_even, dp_odd))

# Call the solve() function
solve()