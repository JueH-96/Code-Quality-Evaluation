import sys

def main():
    n = int(sys.stdin.readline())
    a = list(map(int, sys.stdin.readline().split()))
    
    # Compute left array
    left = [0] * n
    seen = set()
    count = 0
    for i in range(n):
        if a[i] not in seen:
            seen.add(a[i])
            count += 1
        left[i] = count
    
    # Compute right array
    right = [0] * n
    seen = set()
    count = 0
    for i in range(n-1, -1, -1):
        if a[i] not in seen:
            seen.add(a[i])
            count += 1
        right[i] = count
    
    # Find the maximum sum
    max_sum = 0
    for k in range(1, n):
        current = left[k-1] + right[k]
        if current > max_sum:
            max_sum = current
    
    print(max_sum)

if __name__ == "__main__":
    main()