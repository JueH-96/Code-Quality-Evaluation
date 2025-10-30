import sys

def main():
    data = sys.stdin.read().split()
    t = int(data[0])
    index = 1
    results = []
    for _ in range(t):
        n = int(data[index])
        index += 1
        arr = list(map(int, data[index:index+n]))
        index += n
        total = sum(arr)
        cur = 0
        valid = True
        for i in range(n):
            cur += arr[i]
            if cur * n > total * (i + 1):
                valid = False
                break
        results.append("Yes" if valid else "No")
    
    for res in results:
        print(res)

if __name__ == "__main__":
    main()