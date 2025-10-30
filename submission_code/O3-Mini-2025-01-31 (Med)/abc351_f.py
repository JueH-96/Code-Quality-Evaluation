def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return

    n = int(data[0])
    A = list(map(int, data[1:]))

    # We'll compress the values because A_i can be as large as 10^8.
    sorted_unique = sorted(set(A))
    comp = {v: i + 1 for i, v in enumerate(sorted_unique)}  # 1-indexed for BIT
    size = len(sorted_unique)

    # Fenwick trees to store counts and sums respectively.
    BIT_count = [0] * (size + 1)
    BIT_sum = [0] * (size + 1)

    def update(bit, i, delta):
        while i <= size:
            bit[i] += delta
            i += i & -i

    def query(bit, i):
        s = 0
        while i:
            s += bit[i]
            i -= i & -i
        return s

    # For each element A[j], we want to sum the contribution 
    # coming from all previous indices with A[i] < A[j]:
    #   contribution = A[j]*(# of such A[i]) - sum(all such A[i])
    result = 0
    for x in A:
        idx = comp[x]
        count_less = query(BIT_count, idx - 1)
        sum_less = query(BIT_sum, idx - 1)
        result += x * count_less - sum_less
        update(BIT_count, idx, 1)
        update(BIT_sum, idx, x)

    sys.stdout.write(str(result))

if __name__ == '__main__':
    main()