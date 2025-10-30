import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    m = int(data[1])
    a = list(map(int, data[2:2+n]))
    
    a.sort()
    j = 0
    max_count = 0
    for i in range(n):
        while j < n and a[j] < a[i] + m:
            j += 1
        count = j - i
        if count > max_count:
            max_count = count
            
    print(max_count)

if __name__ == "__main__":
    main()