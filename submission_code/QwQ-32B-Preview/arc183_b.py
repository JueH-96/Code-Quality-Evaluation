def can_transform(A, B, N, K):
    from collections import defaultdict
    window = set()
    left = 1
    right = 1
    for i in range(1, N + 1):
        new_left = i - K
        if new_left < 1:
            new_left = 1
        new_right = min(i + K, N)
        
        # Remove elements that are no longer in the window
        while left < new_left:
            if A[left] in window:
                window.remove(A[left])
            left += 1
        while left > new_left:
            left -= 1
            window.add(A[left])
        # Add elements that are new to the window
        while right <= new_right:
            window.add(A[right])
            right += 1
        while right > new_right + 1:
            right -= 1
            if A[right] in window:
                window.remove(A[right])
        
        if B[i] not in window:
            return "No"
    return "Yes"

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    T = int(data[idx])
    idx += 1
    results = []
    for _ in range(T):
        N, K = int(data[idx]), int(data[idx + 1])
        idx += 2
        A = [0] * (N + 1)
        for i in range(1, N + 1):
            A[i] = int(data[idx])
            idx += 1
        B = [0] * (N + 1)
        for i in range(1, N + 1):
            B[i] = int(data[idx])
            idx += 1
        result = can_transform(A, B, N, K)
        results.append(result)
    print('
'.join(results))

if __name__ == "__main__":
    main()