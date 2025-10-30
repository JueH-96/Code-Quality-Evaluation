import sys

def sum_of_absolute_differences(arr):
    """
    Returns the sum over all unordered pairs (i < j) of |arr[i] - arr[j]|.
    Runs in O(len(arr) log len(arr)) time thanks to sorting.
    """
    if len(arr) < 2:
        return 0

    arr.sort()
    prefix_sum = 0
    total = 0
    for i, value in enumerate(arr):
        total += value * i - prefix_sum   # contribution of value as the larger element
        prefix_sum += value
    return total


def main() -> None:
    input_data = sys.stdin.buffer.read().split()
    n = int(input_data[0])
    xs = list(map(int, input_data[1::2]))
    ys = list(map(int, input_data[2::2]))

    # Split the points into the two parity classes of (x + y) % 2
    U_even, V_even = [], []
    U_odd,  V_odd  = [], []

    for x, y in zip(xs, ys):
        u = x + y        # x + y
        v = x - y        # x - y
        if u & 1:        # odd
            U_odd.append(u)
            V_odd.append(v)
        else:            # even
            U_even.append(u)
            V_even.append(v)

    # Sum of |ΔU| and |ΔV| inside each parity class
    total = (
        sum_of_absolute_differences(U_even) +
        sum_of_absolute_differences(U_odd)  +
        sum_of_absolute_differences(V_even) +
        sum_of_absolute_differences(V_odd)
    )

    # Each distance contributes (|ΔU| + |ΔV|) / 2
    answer = total // 2
    print(answer)


if __name__ == "__main__":
    main()