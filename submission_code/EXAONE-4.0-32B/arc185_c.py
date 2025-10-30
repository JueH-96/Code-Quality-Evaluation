import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    x = int(data[1])
    A = list(map(int, data[2:2+n]))
    
    B = [(A[i], i) for i in range(n)]
    B.sort(key=lambda x: x[0])
    
    for i in range(n-2):
        if i > 0 and B[i][0] == B[i-1][0]:
            continue
            
        left = i + 1
        right = n - 1
        target = x - B[i][0]
        
        while left < right:
            total = B[left][0] + B[right][0]
            if total == target:
                indices = sorted([B[i][1], B[left][1], B[right][1]])
                print(f"{indices[0]+1} {indices[1]+1} {indices[2]+1}")
                return
            elif total < target:
                left += 1
                while left < right and B[left][0] == B[left-1][0]:
                    left += 1
            else:
                right -= 1
                while left < right and B[right][0] == B[right+1][0]:
                    right -= 1
                    
    print(-1)

if __name__ == "__main__":
    main()