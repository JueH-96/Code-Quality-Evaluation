import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    k = int(data[1])
    A = list(map(int, data[2:2+n]))
    
    seen = set()
    for num in A:
        if num <= k:
            seen.add(num)
            
    total_sum = k * (k + 1) // 2
    sum_seen = sum(seen)
    result = total_sum - sum_seen
    print(result)

if __name__ == "__main__":
    main()