def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))

    # Coordinate compression
    unique_vals = sorted(set(A))
    comp_index = {}
    for i, val in enumerate(unique_vals):
        comp_index[val] = i + 1  # 1-based index for Fenwick/BIT

    size = len(unique_vals)

    # Fenwicks (BIT) for counts and for sums
    # We'll keep 1-based indexing
    fenwicks_count = [0] * (size + 1)
    fenwicks_sum = [0] * (size + 1)

    # Fenwicks update function
    def fenwicks_update(BIT, idx, value):
        while idx <= size:
            BIT[idx] += value
            idx += idx & -idx

    # Fenwicks query function (sum from 1 to idx)
    def fenwicks_query(BIT, idx):
        s = 0
        while idx > 0:
            s += BIT[idx]
            idx -= idx & -idx
        return s

    answer = 0
    for val in A:
        # Compressed index
        ci = comp_index[val]
        # Query how many previous elements are strictly less than val => fenwicks_query_count(ci - 1)
        c_less = fenwicks_query(fenwicks_count, ci - 1)
        s_less = fenwicks_query(fenwicks_sum, ci - 1)
        answer += val * c_less - s_less
        # Update Fenwicks
        fenwicks_update(fenwicks_count, ci, 1)
        fenwicks_update(fenwicks_sum, ci, val)

    print(answer)

# Call solve() after defining it
if __name__ == "__main__":
    solve()