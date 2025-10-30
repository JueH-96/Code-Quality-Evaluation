import sys

def main():
    input = sys.stdin.read
    data = input().split()
    T = int(data[0])
    idx = 1
    for _ in range(T):
        N, M = int(data[idx]), int(data[idx+1])
        idx +=2
        boxes = []
        for __ in range(N):
            V = int(data[idx])
            P = int(data[idx+1])
            boxes.append(V - P)
            idx +=2
        boxes.sort(reverse=True)
        low = -float('inf')
        high = float('inf')
        ans = 0
        while low <= high:
            mid = (low + high) // 2
            current_sum = 0
            count = 0
            for val in boxes:
                current_sum += val
                if current_sum >= mid:
                    count +=1
                    current_sum = 0
            if count >= M:
                ans = mid
                low = mid +1
            else:
                high = mid -1
        print(ans)

if __name__ == '__main__':
    main()