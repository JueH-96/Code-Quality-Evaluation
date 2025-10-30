import sys

def main():
    n = int(sys.stdin.readline())
    a = list(map(int, sys.stdin.readline().split()))
    
    # Compute left array
    left = [0] * n
    current_set = set()
    for i in range(n):
        current_set.add(a[i])
        left[i] = len(current_set)
    
    # Compute right array
    right = [0] * n
    current_set_right = set()
    for j in range(n-1, -1, -1):
        current_set_right.add(a[j])
        right[j] = len(current_set_right)
    
    # Find the maximum sum
    max_sum = 0
    for i in range(n-1):
        current = left[i] + right[i+1]
        if current > max_sum:
            max_sum = current
    
    print(max_sum)

if __name__ == "__main__":
    main()