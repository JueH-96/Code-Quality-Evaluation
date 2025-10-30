def main():
    import sys
    data = sys.stdin.read().strip().split()
    n = int(data[0])
    a = []
    b = []
    
    # Read input values
    idx = 1
    for _ in range(n):
        ai, bi = int(data[idx]), int(data[idx + 1])
        idx += 2
        a.append(ai)
        b.append(bi)

    # The final height = sum(a) + max(bi - ai).
    # Explanation:
    #   If you choose the ith giant to be on top, the height = sum(a) + (b[i] - a[i]).
    #   So the maximum comes from choosing i that maximizes b[i] - a[i].
    sum_a = sum(a)
    max_diff = max(b[i] - a[i] for i in range(n))
    answer = sum_a + max_diff
    
    print(answer)

# Don't forget to call main()
if __name__ == "__main__":
    main()