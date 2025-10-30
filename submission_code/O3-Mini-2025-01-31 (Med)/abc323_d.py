def main():
    import sys, bisect
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    
    # Create a dictionary to map slime size to its count.
    counts = {}
    sorted_keys = []
    for _ in range(n):
        s = int(next(it))
        c = int(next(it))
        if s in counts:
            counts[s] += c
        else:
            counts[s] = c
            sorted_keys.append(s)
    
    # Sort the sizes so that when we merge slimes,
    # we process increasing sizes.
    sorted_keys.sort()
    
    # Process each size in increasing order.
    # When we merge two slimes of size X, it produces one slime of size 2*X.
    # We combine as many pairs as possible.
    i = 0
    while i < len(sorted_keys):
        size = sorted_keys[i]
        cnt = counts[size]
        pairs = cnt // 2     # number of merges possible at this size
        remainder = cnt % 2   # leftovers that cannot be paired

        # Update the count for the current size to be the leftover.
        counts[size] = remainder

        if pairs > 0:
            new_size = size * 2
            # Add the new slimes (from merges) to the count of new_size.
            if new_size in counts:
                counts[new_size] += pairs
            else:
                counts[new_size] = pairs
                # Maintain sorted order by inserting new_size in sorted_keys.
                bisect.insort(sorted_keys, new_size)
        i += 1

    # The final answer is the total number of remaining slimes,
    # i.e. the sum of counts for all sizes.
    answer = sum(counts.values())
    sys.stdout.write(str(answer))
    
if __name__=="__main__":
    main()