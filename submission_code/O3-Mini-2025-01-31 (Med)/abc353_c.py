def main():
    import sys
    input_data = sys.stdin.read().split()
    n = int(input_data[0])
    A = list(map(int, input_data[1:]))
    mod_val = 100_000_000

    # The sum over all pairs (i, j) with i < j of (A_i + A_j) equals (n-1) * sum(A)
    total_sum = (n - 1) * sum(A)

    # However, the function f(x, y) returns (x+y) mod mod_val,
    # and note that for each pair (i, j) such that A_i + A_j >= mod_val,
    # the mod function subtracts mod_val from (A_i + A_j). 
    # Thus, we need to subtract mod_val for each such pair.
    
    # Sort array to count pairs with A_i + A_j >= mod_val efficiently.
    A.sort()
    count_pairs = 0
    left = 0
    right = n - 1

    while left < right:
        if A[left] + A[right] >= mod_val:
            count_pairs += (right - left)
            right -= 1
        else:
            left += 1

    # The answer is the unmodified total sum minus mod_val for each "wrap-around" pair.
    answer = total_sum - mod_val * count_pairs
    sys.stdout.write(str(answer))
    
if __name__ == '__main__':
    main()