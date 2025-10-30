import sys


def main() -> None:
    data = sys.stdin.buffer.read().split()
    if not data:                 # empty input guard
        return
    n = int(data[0])
    a = list(map(int, data[1:1 + n]))

    s = sum(a)                   # total sum
    x = s // n                   # value of the “lower” group
    k = s - x * n                # how many elements must be x+1

    low_val = x
    high_val = x + 1

    total_low_cost = 0           # Σ |A_i - low_val|
    diffs = []                   # cost_high - cost_low for every index

    for v in a:
        cost_low = abs(v - low_val)
        total_low_cost += cost_low
        diffs.append(abs(v - high_val) - cost_low)

    # choose k indices with smallest extra cost if they are set to high_val
    if k:
        diffs.sort()
        total_cost = total_low_cost + sum(diffs[:k])
    else:
        total_cost = total_low_cost

    # every operation removes a unit from a surplus index
    # and gives it to a deficit index → divides total L1–distance by 2
    answer = total_cost // 2
    print(answer)


if __name__ == "__main__":
    main()