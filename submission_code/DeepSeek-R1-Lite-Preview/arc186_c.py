def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    T = int(data[idx])
    idx += 1
    results = []
    for _ in range(T):
        N = int(data[idx])
        M = int(data[idx+1])
        idx += 2
        boxes = []
        for _ in range(N):
            V = int(data[idx])
            P = int(data[idx+1])
            idx += 2
            boxes.append(V - P)
        # Sort the boxes by (V - P) in descending order
        boxes.sort(reverse=True)
        # Binary search for the maximum X such that at least M boxes have (V - P) >= X
        left = -10**18
        right = 10**18
        answer = 0
        while left <= right:
            mid = (left + right) // 2
            # Count the number of boxes with (V - P) >= mid
            count = 0
            for profit in boxes:
                if profit >= mid:
                    count += 1
                else:
                    break  # since sorted descending
            if count >= M:
                answer = mid
                left = mid + 1
            else:
                right = mid - 1
        results.append(str(answer))
    print("
".join(results))

if __name__ == "__main__":
    main()