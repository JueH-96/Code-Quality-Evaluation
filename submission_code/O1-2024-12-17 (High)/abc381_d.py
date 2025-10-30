def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # Build array E for pairs starting at even indices: (A[0],A[1]), (A[2],A[3]), ...
    # E[i] will store the integer of that pair if A[2i] == A[2i+1], otherwise None.
    E = []
    for i in range(N // 2):
        if A[2*i] == A[2*i + 1]:
            E.append(A[2*i])
        else:
            E.append(None)

    # Build array O for pairs starting at odd indices: (A[1],A[2]), (A[3],A[4]), ...
    # O[i] will store the integer of that pair if A[2i+1] == A[2i+2], otherwise None.
    O = []
    for i in range((N - 1) // 2):
        if A[2*i + 1] == A[2*i + 2]:
            O.append(A[2*i + 1])
        else:
            O.append(None)

    # A helper function to find the longest contiguous segment of arr
    # that has no None and all distinct elements (i.e., no repeats).
    # The length returned is in "pair units" (number of valid pairs).
    def longest_distinct_non_none(arr):
        used = set()
        left = 0
        max_len = 0
        for right in range(len(arr)):
            # If arr[right] is None, we cannot use this index at all:
            # reset the window.
            if arr[right] is None:
                used.clear()
                left = right + 1
            else:
                # Remove from the left until arr[right] can be added distinctively
                while arr[right] in used and left <= right:
                    used.remove(arr[left])
                    left += 1
                used.add(arr[right])
                max_len = max(max_len, right - left + 1)
        return max_len

    ansE = longest_distinct_non_none(E)
    ansO = longest_distinct_non_none(O)
    # Multiply by 2 because each "pair unit" corresponds to 2 elements in the original array.
    print(max(ansE, ansO) * 2)

# Do not forget to call main()
main()