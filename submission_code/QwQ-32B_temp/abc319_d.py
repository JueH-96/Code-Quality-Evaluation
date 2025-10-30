import sys

def main():
    N, M = map(int, sys.stdin.readline().split())
    L = list(map(int, sys.stdin.readline().split()))
    
    max_L = max(L)
    sum_L = sum(L)
    upper = sum_L + (N - 1)
    
    left = max_L
    right = upper
    
    def is_possible(W):
        lines = 1
        current = 0
        for l in L:
            if current == 0:
                next_width = l
            else:
                next_width = current + 1 + l
            if next_width <= W:
                current = next_width
            else:
                lines += 1
                current = l
            if lines > M:
                return False
        return True
    
    while left < right:
        mid = (left + right) // 2
        if is_possible(mid):
            right = mid
        else:
            left = mid + 1
    print(left)

if __name__ == "__main__":
    main()