import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    k = int(data[1])
    A = list(map(int, data[2:2+n]))
    
    total = k * (k + 1) // 2
    seen = set()
    subtract_sum = 0
    
    for num in A:
        if num <= k and num not in seen:
            seen.add(num)
            subtract_sum += num
            
    result = total - subtract_sum
    print(result)

if __name__ == "__main__":
    main()