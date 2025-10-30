import sys

def main():
    n, m = map(int, sys.stdin.readline().split())
    sum_x = 0
    current = 1
    max_sum = 10**9
    
    for _ in range(m + 1):
        sum_x += current
        if sum_x > max_sum:
            print("inf")
            return
        current *= n
    
    print(sum_x)

if __name__ == "__main__":
    main()