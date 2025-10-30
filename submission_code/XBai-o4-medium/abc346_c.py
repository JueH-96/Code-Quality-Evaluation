import sys

def main():
    n, k = map(int, sys.stdin.readline().split())
    a = list(map(int, sys.stdin.readline().split()))
    
    present = set()
    for num in a:
        if 1 <= num <= k:
            present.add(num)
    
    total = k * (k + 1) // 2
    sum_present = sum(present)
    print(total - sum_present)

if __name__ == "__main__":
    main()