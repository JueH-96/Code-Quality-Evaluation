import sys

def main():
    n, m = map(int, sys.stdin.readline().split())
    L = list(map(int, sys.stdin.readline().split()))
    max_L = max(L)
    sum_total = sum(L) + (n - 1)
    
    if m >= n:
        print(max_L)
        return
    
    low = max_L
    high = sum_total
    answer = sum_total
    
    while low <= high:
        mid = (low + high) // 2
        required = 1
        current = 0
        for num in L:
            if current == 0:
                current = num
            else:
                if current + 1 + num > mid:
                    required += 1
                    current = num
                else:
                    current += 1 + num
            if required > m:
                break
        if required <= m:
            answer = mid
            high = mid - 1
        else:
            low = mid + 1
    print(answer)

if __name__ == "__main__":
    main()