def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    a = list(map(int, input[1:n+1]))
    
    # Compute left array
    left = []
    unique_left = set()
    for num in a:
        unique_left.add(num)
        left.append(len(unique_left))
    
    # Compute right array
    right = [0] * n
    unique_right = set()
    for i in range(n-1, -1, -1):
        unique_right.add(a[i])
        right[i] = len(unique_right)
    
    # Find maximum sum of left[i] + right[i+1]
    max_sum = 0
    for i in range(n-1):
        current = left[i] + right[i+1]
        if current > max_sum:
            max_sum = current
    print(max_sum)

if __name__ == "__main__":
    main()