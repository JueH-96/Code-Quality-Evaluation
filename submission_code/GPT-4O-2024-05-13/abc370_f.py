# YOUR CODE HERE
def can_divide(A, K, min_sum):
    current_sum = 0
    count = 1
    for mass in A:
        if current_sum + mass > min_sum:
            count += 1
            current_sum = mass
            if count > K:
                return False
        else:
            current_sum += mass
    return True

def find_min_max_sum(A, K):
    left, right = max(A), sum(A)
    while left < right:
        mid = (left + right) // 2
        if can_divide(A, K, mid):
            right = mid
        else:
            left = mid + 1
    return left

def count_uncut_lines(A, K, min_sum):
    current_sum = 0
    count = 0
    for mass in A:
        if current_sum + mass > min_sum:
            count += 1
            current_sum = mass
        else:
            current_sum += mass
    return len(A) - count

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:]))
    
    min_sum = find_min_max_sum(A, K)
    uncut_lines = count_uncut_lines(A, K, min_sum)
    
    print(min_sum, uncut_lines)

if __name__ == "__main__":
    main()