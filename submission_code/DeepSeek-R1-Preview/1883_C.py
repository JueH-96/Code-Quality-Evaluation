import sys

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    t = int(data[idx])
    idx += 1
    for _ in range(t):
        n, k = int(data[idx]), int(data[idx+1])
        idx += 2
        a = list(map(int, data[idx:idx+n]))
        idx += n
        
        rem = 1
        for num in a:
            rem = (rem * (num % k)) % k
        
        if rem == 0:
            print(0)
            continue
        
        min_step = min((k - (num % k)) for num in a)
        print(min_step)
        
if __name__ == "__main__":
    main()