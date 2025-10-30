# YOUR CODE HERE
def merge_sort_count(arr):
    # returns sorted array and inversion count
    n = len(arr)
    if n <= 1:
        return arr, 0
    mid = n >> 1
    left, invL = merge_sort_count(arr[:mid])
    right, invR = merge_sort_count(arr[mid:])
    merged = []
    inv = invL + invR
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            inv += len(left) - i
            j += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged, inv

def count_inversions(arr):
    # returns inversion count of arr
    return merge_sort_count(arr)[1]

def bubble_sort_pass(P, k):
    # simulate one bubble sort pass on the prefix P[0:k]
    # we use an optimization: if a contiguous descending run is detected,
    # we “rotate” it by one (this is exactly what the pass does).
    swaps = 0
    i = 0
    # Process indices 0 .. k-2
    while i < k - 1:
        if P[i] <= P[i+1]:
            i += 1
        else:
            # found an inversion so determine the maximal block [i, j)
            j = i + 1
            while j < k and P[j-1] > P[j]:
                j += 1
            run_length = j - i
            # In this block, one left–rotation fixes all adjacent inversions.
            swaps += run_length - 1
            # rotate the block: [P[i], P[i+1], …, P[j-1]] becomes [P[i+1], …, P[j-1], P[i]]
            P[i:j] = P[i+1:j] + P[i:i+1]
            # then continue scanning from j-1
            i = j - 1
    return swaps

def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    # Read permutation P – a list of n integers.
    P = [int(next(it)) for _ in range(n)]
    m = int(next(it))
    # Read the M queries (they are non-decreasing and in the range 2..n)
    queries = [int(next(it)) for _ in range(m)]
    
    # Compute the inversion count of the original permutation.
    inv0 = count_inversions(P)
    global_inv = inv0

    out_lines = []
    # For every query (operation), perform one bubble-sort pass on prefix P[0:k]
    for k in queries:
        sw = bubble_sort_pass(P, k)
        global_inv -= sw  # each swap fixes one inversion.
        out_lines.append(str(global_inv))
    sys.stdout.write("
".join(out_lines))

if __name__ == '__main__':
    main()