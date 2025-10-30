def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    A = list(map(int, data[1:]))

    # Explanation:
    # At each operation, we remove one from two positive elements.
    # Thus, the total sum decreases by 2. 
    # However, if one element is so large that the rest cannot pair with it,
    # then the maximum operations is limited by the sum of the rest.
    # Hence, the number of operations is:
    #   min(total//2, total - max(A)),
    # where total is the sum of all elements.
    
    total = sum(A)
    max_value = max(A)
    result = min(total // 2, total - max_value)
    print(result)

if __name__ == '__main__':
    main()