import sys

def main() -> None:
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    n = int(data[0])
    a = list(map(int, data[1:n + 1]))

    # Pair each value with its index and sort descending by value
    pairs = [(a[i], i) for i in range(n)]
    pairs.sort(key=lambda x: -x[0])

    res = [0] * n           # answers to output
    sum_so_far = 0          # sum of elements already processed (strictly larger)

    i = 0
    while i < n:
        v = pairs[i][0]     # current value
        j = i
        group_sum = 0       # total of this equal-value group

        # process all elements that have the same value v
        while j < n and pairs[j][0] == v:
            idx = pairs[j][1]
            res[idx] = sum_so_far   # all previously processed elements are greater than v
            group_sum += v
            j += 1

        sum_so_far += group_sum     # now these elements become "greater" for the next groups
        i = j

    print(' '.join(map(str, res)))

if __name__ == "__main__":
    main()