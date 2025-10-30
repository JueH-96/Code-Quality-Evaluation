def solve():
    n = int(input())
    sticks = [input() for _ in range(n)]

    counted_sticks = set()
    different_sticks_count = 0

    for stick in sticks:
        reversed_stick = stick[::-1]
        canonical_stick = min(stick, reversed_stick)
        if canonical_stick not in counted_sticks:
            different_sticks_count += 1
            counted_sticks.add(canonical_stick)

    print(different_sticks_count)

solve()