def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    arr = list(map(int, data[1:]))
    # Total number of subarrays.
    total_intervals = n * (n + 1) // 2

    # For each distinct value, we can count the number of subarrays that contain it.
    # Then, the required answer is the sum over all distinct values of 
    # (subarrays that contain that value). This is valid because in each subarray,
    # each distinct value contributes exactly once to f(l, r).

    # For a particular value v, suppose its positions in the array are p1, p2, …, pk.
    # Let p0 = 0 and p{k+1} = n+1. The intervals that do NOT contain v are those 
    # which fall entirely into one of the gaps between these positions. In the gap 
    # between p_i and p_{i+1}, the number of subarrays is:
    #    gap_size = (p_{i+1} - p_i - 1)
    #    subarrays in that gap = gap_size * (gap_size + 1) // 2
    # Then, the number of subarrays that contain at least one occurrence of v is:
    #    total_intervals - (sum of gap subarrays)
    
    from collections import defaultdict
    positions = defaultdict(list)
    for i, value in enumerate(arr, start=1):
        positions[value].append(i)
    
    result = 0
    for value, pos_list in positions.items():
        # Add boundaries to simplify gap computations
        pos_list = [0] + pos_list + [n + 1]
        missing = 0
        for i in range(len(pos_list) - 1):
            gap = pos_list[i + 1] - pos_list[i] - 1
            missing += gap * (gap + 1) // 2
        # Subarrays containing value `value`
        count = total_intervals - missing
        result += count
    
    sys.stdout.write(str(result))


if __name__ == "__main__":
    main()